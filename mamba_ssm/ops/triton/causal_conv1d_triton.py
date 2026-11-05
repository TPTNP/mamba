# Copyright (c) 2026, TPTNP.
"""causal_conv1d's forward and backward for Mamba2's fused path, in Triton.

Mamba2 has a fused path -- one autograd function over the split of its input projection, the
depthwise causal convolution, the scan, the gated norm and the output projection -- and an
unfused one that runs each as its own operation.  The fused path calls the ``causal_conv1d``
package's compiled forward and backward, and that package is CUDA C++, so on an Intel GPU Mamba2
had to take the unfused path.  There the convolution went through oneDNN with a layout reorder
each way, and the backward of the projection's split was a concatenation of its full width: on an
Aurora tile some 250 ms of a 1.7 s training step, against a few milliseconds of real work.

These two functions take causal_conv1d's place with its signatures, as Triton kernels that run
wherever Mamba2's other kernels do.  ``ssd_combined`` uses them when the package is absent, and in
its place with MAMBA_TRITON_CONV=1: on an RTX 5090 they were the faster (554x1280x96 bf16,
backward 0.46 against 0.59 ms, forward 0.18 against 0.19).  Like causal_conv1d's channels-last
kernels they read the activations where they lie, a strided slice of the input projection, and
the backward writes dx straight into the slice of the projection's gradient the fused path hands
it.  The weight and bias gradients are summed in a fixed order, so they come out the same every
time.  The tile shape is tuned per device: a 64x64 float32 tile over four warps suits an NVIDIA
card, but on an Aurora tile it is half an Xe thread's registers, and the backward holds several --
the first build ran its backward at 15 ms a call against a 0.6 ms floor.  Where Triton cannot run
-- on the CPU, or if a kernel fails to build -- the same convolution runs as PyTorch operations,
with autograd's backward.

Whole sequences only: a ``seq_idx`` or a carried state raises rather than being ignored.
"""
from __future__ import annotations

import os,warnings

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:triton=None

from mamba_ssm.utils.device import device_context


def _on_intel_gpu():
    xpu=getattr(torch,"xpu",None)
    return xpu is not None and xpu.is_available() and not torch.cuda.is_available()


def _configs(tiles,names):
    """Triton configurations for ``tiles``, each the values of ``names`` and then a warp count; on an
    Intel GPU every tile is also tried at 8 and 16 warps, since four, which suits an NVIDIA card,
    leaves most of an Xe core idle."""
    extra=(8,16) if _on_intel_gpu() else ();seen=set();configs=[]
    for *values,warps in tiles:
        for count in (warps,*extra):
            if (tuple(values),count) in seen:continue
            seen.add((tuple(values),count));configs.append(triton.Config(dict(zip(names,values)),num_warps=count))
    return configs


# What the tuner chooses among.  The forward's (positions, channels, warps); the first suits an
# NVIDIA card, the rest give each work-item less of the tile, which Intel's register files need.
FORWARD_TILES=((64,64,4),(32,64,4),(16,64,4),(32,128,8),(16,128,8),(64,64,16),(32,64,16),(16,64,16))
# The backward's (channels, warps).  Its positions per tile stay fixed, since the weight
# gradient is summed one row per tile and the rows must be counted before the tuner chooses.
BACKWARD_LENGTH=32
BACKWARD_TILES=((64,4),(32,4),(128,4),(64,8),(32,8),(128,8),(64,16),(32,16),(128,16))
# Width 4's (positions, channels, warps): it sums the weight gradient along each sequence, so its
# positions per step are the tuner's to choose too.  The first was the 5090's best (0.46 ms at the
# xlarge micro-batch's 554x1280x96); fewer positions a step hold fewer values in registers, which
# is what an Intel GPU's need.  An Aurora tile chose (16, 64, 8): 1.76 ms, from 2.70 with the sums
# reduced at every step instead of once a sequence.
BACKWARD_TILES4=((32,32,2),*((positions,channels,warps) for positions in (8,16,32,64) for channels in (32,64,128) for warps in (2,4)
    if (positions,channels,warps)!=(32,32,2)))
_triton={"working":triton is not None}


def _convolve(x,weight,bias,silu):
    """Depthwise causal convolution of ``x`` (batch, length, channels) by ``weight`` (channels, width).

    Output position t reads positions t-width+1 .. t, the last weight on the current one, as
    causal_conv1d and Mamba2's own padded ``Conv1d`` both do.  At least float32 throughout, one
    rounding back to the input's dtype.  The reference the kernels are tested against.
    """
    width=weight.shape[-1];length=x.shape[1];wide=torch.promote_types(x.dtype,torch.float32)
    padded=F.pad(x.to(wide),(0,0,width-1,0))
    out=padded[:,0:length]*weight[:,0].to(wide)
    for k in range(1,width):out=out+padded[:,k:k+length]*weight[:,k].to(wide)
    if bias is not None:out=out+bias.to(wide)
    return (F.silu(out) if silu else out).to(x.dtype)


if triton is not None:
    @triton.autotune(configs=_configs(FORWARD_TILES,("BLOCK_L","BLOCK_C")),key=["channels"])
    @triton.jit
    def _forward_kernel(x,weight,bias,out,length,channels,
                        x_batch,x_channel,x_position,out_batch,out_channel,out_position,weight_channel,weight_tap,
                        HAS_BIAS:tl.constexpr,SILU:tl.constexpr,WIDTH:tl.constexpr,BLOCK_L:tl.constexpr,BLOCK_C:tl.constexpr):
        batch=tl.program_id(2).to(tl.int64)
        t=tl.program_id(0)*BLOCK_L+tl.arange(0,BLOCK_L)
        c=tl.program_id(1)*BLOCK_C+tl.arange(0,BLOCK_C)
        inside=c<channels
        acc=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
        if HAS_BIAS:acc+=tl.load(bias+c,mask=inside,other=0.0).to(tl.float32)[None,:]
        for k in tl.static_range(WIDTH):
            source=t-(WIDTH-1)+k
            present=(source>=0)&(source<length)
            value=tl.load(x+batch*x_batch+source[:,None]*x_position+c[None,:]*x_channel,mask=present[:,None]&inside[None,:],other=0.0)
            tap=tl.load(weight+c*weight_channel+k*weight_tap,mask=inside,other=0.0)
            acc+=value.to(tl.float32)*tap.to(tl.float32)[None,:]
        if SILU:acc=acc*tl.sigmoid(acc)
        tl.store(out+batch*out_batch+t[:,None]*out_position+c[None,:]*out_channel,acc.to(out.dtype.element_ty),
                 mask=(t<length)[:,None]&inside[None,:])

    @triton.autotune(configs=_configs(BACKWARD_TILES,("BLOCK_C",)),key=["channels"])
    @triton.jit
    def _backward_kernel(x,weight,bias,dout,dx,dweight_tiles,dbias_tiles,length,channels,
                         x_batch,x_channel,x_position,dout_batch,dout_channel,dout_position,dx_batch,dx_channel,dx_position,
                         weight_channel,weight_tap,
                         HAS_BIAS:tl.constexpr,SILU:tl.constexpr,WIDTH:tl.constexpr,BLOCK_L:tl.constexpr,BLOCK_C:tl.constexpr):
        """dx[t] = sum_k w[k] g[t+width-1-k]; dw[k] and db summed over this tile's own positions.

        g is the gradient at the convolution's output before the SiLU: dout times SiLU's slope,
        which needs the pre-activation recomputed at each of the width positions dx reads.
        """
        batch=tl.program_id(2).to(tl.int64)
        t=tl.program_id(0)*BLOCK_L+tl.arange(0,BLOCK_L)
        c=tl.program_id(1)*BLOCK_C+tl.arange(0,BLOCK_C)
        inside=c<channels
        tile=batch*tl.num_programs(0)+tl.program_id(0)
        gathered=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
        for j in tl.static_range(WIDTH):
            s=t+j
            g=tl.load(dout+batch*dout_batch+s[:,None]*dout_position+c[None,:]*dout_channel,
                      mask=(s<length)[:,None]&inside[None,:],other=0.0).to(tl.float32)
            if SILU:
                pre=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
                if HAS_BIAS:pre+=tl.load(bias+c,mask=inside,other=0.0).to(tl.float32)[None,:]
                for k in tl.static_range(WIDTH):
                    source=s-(WIDTH-1)+k
                    present=(source>=0)&(source<length)
                    value=tl.load(x+batch*x_batch+source[:,None]*x_position+c[None,:]*x_channel,mask=present[:,None]&inside[None,:],other=0.0)
                    pre+=value.to(tl.float32)*tl.load(weight+c*weight_channel+k*weight_tap,mask=inside,other=0.0).to(tl.float32)[None,:]
                sigmoid=tl.sigmoid(pre)
                g=g*sigmoid*(1.0+pre*(1.0-sigmoid))
            tap=tl.load(weight+c*weight_channel+(WIDTH-1-j)*weight_tap,mask=inside,other=0.0).to(tl.float32)
            gathered+=g*tap[None,:]
            if j==0:
                for k in tl.static_range(WIDTH):
                    source=t-(WIDTH-1)+k
                    present=(source>=0)&(source<length)
                    value=tl.load(x+batch*x_batch+source[:,None]*x_position+c[None,:]*x_channel,mask=present[:,None]&inside[None,:],other=0.0)
                    tl.store(dweight_tiles+(tile*WIDTH+k)*channels+c,tl.sum(g*value.to(tl.float32),axis=0),mask=inside)
                if HAS_BIAS:tl.store(dbias_tiles+tile*channels+c,tl.sum(g,axis=0),mask=inside)
        tl.store(dx+batch*dx_batch+t[:,None]*dx_position+c[None,:]*dx_channel,gathered.to(dx.dtype.element_ty),
                 mask=(t<length)[:,None]&inside[None,:])


def _grid(x,length=None):
    batch,channels,positions=x.shape
    return lambda meta:(triton.cdiv(positions,length or meta["BLOCK_L"]),triton.cdiv(channels,meta["BLOCK_C"]),batch)


def _kernel_forward(x,weight,bias,silu,tile=None):
    """The tuned forward, or with ``tile`` = (positions, channels, warps) that configuration alone."""
    out=torch.empty_like(x)
    kernel,chosen=(_forward_kernel,{}) if tile is None else (_forward_kernel.fn,dict(BLOCK_L=tile[0],BLOCK_C=tile[1],num_warps=tile[2]))
    with device_context(x.device):
        kernel[_grid(x)](x,weight,bias if bias is not None else weight,out,x.shape[2],x.shape[1],
                         x.stride(0),x.stride(1),x.stride(2),out.stride(0),out.stride(1),out.stride(2),
                         weight.stride(0),weight.stride(1),
                         HAS_BIAS=bias is not None,SILU=silu,WIDTH=weight.shape[1],**chosen)
    return out


if triton is not None:
    @triton.autotune(configs=_configs(BACKWARD_TILES4,("BLOCK_L","BLOCK_C")),key=["channels"])
    @triton.jit
    def _backward_kernel4(x,weight,bias,dout,dx,dweight_rows,dbias_rows,length,channels,
                          x_batch,x_channel,x_position,dout_batch,dout_channel,dout_position,dx_batch,dx_channel,dx_position,
                          weight_channel,weight_tap,
                          HAS_BIAS:tl.constexpr,SILU:tl.constexpr,BLOCK_L:tl.constexpr,BLOCK_C:tl.constexpr):
        """The width-4 backward: a program walks one sequence's channels along its length, every tile
        it reads loaded once -- x at t-3 .. t+3 and dout at t .. t+3 -- and its weight and bias
        gradients summed as it goes, one row a sequence.

        The general kernel recomputes the pre-activation at each of the four places dx reads by
        loading x afresh each time, 24 tile loads where 11 are distinct; on an Aurora tile it was
        load-bound, 3.1 ms a call against 0.54 for the forward's 4.  A window of four x tiles slides
        along t, each loaded just before its first use and dropped after its last, so at most seven
        tiles are live at once (Intel's register file is small).  That brought a tile to 2.39 ms; the
        positions a step took were still fixed at 32, since the weight gradient was stored one row
        per tile and the rows had to be counted before the tuner chose.  Summed along the walk
        instead, the rows are the sequences and the step is the tuner's.  The same arithmetic in the
        same order for dx: the pre-activation summed from the bias on, tap by tap, and dx from g at
        t on.

        The sums are kept as products added elementwise, one running tile per tap and the bias, and
        reduced across positions once at the end of the walk.  Reduced at every step, they were the
        larger part of an Aurora tile's 2.63 ms: with them switched off the kernel took 1.62, with them
        and the SiLU recompute off 0.38 (scripts/causal_conv_bench.py --ablate) -- a reduction across
        a tile's rows costs an Intel GPU a trip through shared memory, five of them every step.
        """
        batch=tl.program_id(1).to(tl.int64)
        c=tl.program_id(0)*BLOCK_C+tl.arange(0,BLOCK_C)
        inside=c<channels
        rows=x+batch*x_batch+c[None,:]*x_channel
        grads=dout+batch*dout_batch+c[None,:]*dout_channel
        w0=tl.load(weight+c*weight_channel,mask=inside,other=0.0).to(tl.float32)[None,:]
        w1=tl.load(weight+c*weight_channel+weight_tap,mask=inside,other=0.0).to(tl.float32)[None,:]
        w2=tl.load(weight+c*weight_channel+2*weight_tap,mask=inside,other=0.0).to(tl.float32)[None,:]
        w3=tl.load(weight+c*weight_channel+3*weight_tap,mask=inside,other=0.0).to(tl.float32)[None,:]
        if HAS_BIAS:shift=tl.load(bias+c,mask=inside,other=0.0).to(tl.float32)[None,:]
        dw0=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32);dw1=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
        dw2=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32);dw3=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
        db=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
        for start in range(0,length,BLOCK_L):
            t=start+tl.arange(0,BLOCK_L)
            a=tl.load(rows+(t-3)[:,None]*x_position,mask=((t-3>=0)&(t-3<length))[:,None]&inside[None,:],other=0.0).to(tl.float32)
            b=tl.load(rows+(t-2)[:,None]*x_position,mask=((t-2>=0)&(t-2<length))[:,None]&inside[None,:],other=0.0).to(tl.float32)
            d=tl.load(rows+(t-1)[:,None]*x_position,mask=((t-1>=0)&(t-1<length))[:,None]&inside[None,:],other=0.0).to(tl.float32)
            gathered=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
            for j in tl.static_range(4):
                # The window holds x at t+j-3 .. t+j as a, b, d, e.
                s=t+j
                e=tl.load(rows+s[:,None]*x_position,mask=(s<length)[:,None]&inside[None,:],other=0.0).to(tl.float32)
                g=tl.load(grads+s[:,None]*dout_position,mask=(s<length)[:,None]&inside[None,:],other=0.0).to(tl.float32)
                if SILU:
                    pre=tl.zeros((BLOCK_L,BLOCK_C),dtype=tl.float32)
                    if HAS_BIAS:pre+=shift
                    pre+=a*w0;pre+=b*w1;pre+=d*w2;pre+=e*w3
                    sigmoid=tl.sigmoid(pre)
                    g=g*sigmoid*(1.0+pre*(1.0-sigmoid))
                if j==0:
                    dw0+=g*a;dw1+=g*b;dw2+=g*d;dw3+=g*e
                    if HAS_BIAS:db+=g
                    gathered+=g*w3
                elif j==1:gathered+=g*w2
                elif j==2:gathered+=g*w1
                else:gathered+=g*w0
                a=b;b=d;d=e
            tl.store(dx+batch*dx_batch+t[:,None]*dx_position+c[None,:]*dx_channel,gathered.to(dx.dtype.element_ty),
                     mask=(t<length)[:,None]&inside[None,:])
        tl.store(dweight_rows+(batch*4+0)*channels+c,tl.sum(dw0,axis=0),mask=inside);tl.store(dweight_rows+(batch*4+1)*channels+c,tl.sum(dw1,axis=0),mask=inside)
        tl.store(dweight_rows+(batch*4+2)*channels+c,tl.sum(dw2,axis=0),mask=inside);tl.store(dweight_rows+(batch*4+3)*channels+c,tl.sum(dw3,axis=0),mask=inside)
        if HAS_BIAS:tl.store(dbias_rows+batch*channels+c,tl.sum(db,axis=0),mask=inside)


def _kernel_backward(x,weight,bias,dout,dx,silu,tile=None):
    """The tuned backward, or with ``tile`` that configuration alone: (positions, channels, warps)
    at width 4, (channels, warps) at any other.

    Width 4, Mamba2's, takes the kernel that loads each tile once and walks each sequence; any
    other the general one.
    """
    if dx is None:dx=torch.empty_like(x)
    channels=x.shape[1];width=weight.shape[1];strides=(x.stride(0),x.stride(1),x.stride(2),dout.stride(0),dout.stride(1),dout.stride(2),
        dx.stride(0),dx.stride(1),dx.stride(2),weight.stride(0),weight.stride(1))
    if width==4:
        dweight_rows=torch.empty(x.shape[0],4,channels,device=x.device,dtype=torch.float32)
        dbias_rows=torch.empty(x.shape[0],channels,device=x.device,dtype=torch.float32) if bias is not None else dweight_rows
        kernel,chosen=(_backward_kernel4,{}) if tile is None else (_backward_kernel4.fn,dict(BLOCK_L=tile[0],BLOCK_C=tile[1],num_warps=tile[2]))
        with device_context(x.device):
            kernel[lambda meta:(triton.cdiv(channels,meta["BLOCK_C"]),x.shape[0])](x,weight,bias if bias is not None else weight,dout,dx,
                dweight_rows,dbias_rows,x.shape[2],channels,*strides,HAS_BIAS=bias is not None,SILU=silu,**chosen)
    else:
        tiles=triton.cdiv(x.shape[2],BACKWARD_LENGTH)*x.shape[0]
        dweight_rows=torch.empty(tiles,width,channels,device=x.device,dtype=torch.float32)
        dbias_rows=torch.empty(tiles,channels,device=x.device,dtype=torch.float32) if bias is not None else dweight_rows
        kernel,chosen=(_backward_kernel,{}) if tile is None else (_backward_kernel.fn,dict(BLOCK_C=tile[0],num_warps=tile[1]))
        with device_context(x.device):
            kernel[_grid(x,BACKWARD_LENGTH)](x,weight,bias if bias is not None else weight,dout,dx,dweight_rows,dbias_rows,x.shape[2],channels,
                *strides,HAS_BIAS=bias is not None,SILU=silu,WIDTH=width,BLOCK_L=BACKWARD_LENGTH,**chosen)
    dweight=dweight_rows.sum(0).t().contiguous()
    return dx,dweight,(dbias_rows.sum(0) if bias is not None else None)


def _torch_forward(x,weight,bias,silu):
    return _convolve(x.transpose(1,2),weight,bias,silu).transpose(1,2)


def _torch_backward(x,weight,bias,dout,dx,silu):
    with torch.enable_grad():
        inputs=[x.transpose(1,2).detach().requires_grad_(),weight.detach().requires_grad_()]
        if bias is not None:inputs.append(bias.detach().requires_grad_())
        out=_convolve(inputs[0],inputs[1],inputs[2] if bias is not None else None,silu)
        grads=torch.autograd.grad(out,inputs,dout.transpose(1,2))
    gradient=grads[0].transpose(1,2)
    if dx is None:dx=gradient
    else:dx.copy_(gradient)
    return dx,grads[1],(grads[2] if bias is not None else None)


def _dispatch(kernel,fallback,x,*arguments):
    """The Triton kernel on a GPU; PyTorch on the CPU, or for good once a kernel has failed to build."""
    if _triton["working"] and x.device.type!="cpu":
        try:return kernel(x,*arguments)
        except Exception as error:
            _triton["working"]=False
            warnings.warn(f"the portable causal convolution's Triton kernel failed; running it in PyTorch from now on: {error}")
    return fallback(x,*arguments)


def _unsupported(seq_idx,*states):
    if seq_idx is not None or any(state is not None and state is not False for state in states):
        raise NotImplementedError("the portable causal convolution covers whole sequences only: no seq_idx, no carried states")


def causal_conv1d_fwd_function(x,weight,bias,seq_idx,initial_states,final_states_out,silu_activation):
    """causal_conv1d's forward: ``x`` is (batch, channels, length), returned in the same shape and layout."""
    _unsupported(seq_idx,initial_states,final_states_out)
    with torch.no_grad():return _dispatch(_kernel_forward,_torch_forward,x,weight,bias,bool(silu_activation))


def causal_conv1d_bwd_function(x,weight,bias,dout,seq_idx,initial_states,dfinal_states,dx,return_dinitial_states,silu_activation):
    """causal_conv1d's backward: (dx, dweight, dbias, None), with dx written into ``dx`` when given.

    The fused path hands in ``dx`` as its slice of the input projection's gradient and reads the
    answer from there, so it is filled in place rather than replaced.
    """
    _unsupported(seq_idx,initial_states,dfinal_states,return_dinitial_states)
    with torch.no_grad():
        dx,dweight,dbias=_dispatch(_kernel_backward,_torch_backward,x,weight,bias,dout,dx,bool(silu_activation))
    return dx,dweight.to(weight.dtype),(dbias.to(bias.dtype) if bias is not None else None),None


def preferred():
    """Whether the fused path takes these even where causal_conv1d is installed: MAMBA_TRITON_CONV=1."""
    return os.environ.get("MAMBA_TRITON_CONV","0")=="1"
