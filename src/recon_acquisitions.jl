using FFTW
using Gadgetron
using Gadgetron.MRD
using Gadgetron.Stream
using Gadgetron.Default 
using PartialFunctions
using LinearAlgebra
using Setfield

fft_scaling(xsize,dims=[1]) = sqrt(reduce((*),map(d->xsize[d],dims)))

ortho_fft(x,dims=[1]) = fft(x,dims) ./ Float32(fft_scaling(size(x),dims))

ortho_ifft(x,dims=[1]) = bfft(x,dims) ./ Float32(fft_scaling(size(x),dims))

cfft(x, dims=[1]) = ifftshift(ortho_fft(fftshift(x,dims),dims),dims)


cifft(x, dims=[1]) = fftshift(ortho_ifft(ifftshift(x,dims),dims),dims)


function noise_adjustment(header::MRDHeader, noise_acq::Acquisition)


    noise_bandwidth = header.acquisitionSystemInformation.relativeReceiverNoiseBandwidth |>   default(0.793f0)


    function whitening_transform(noise::Array{ComplexF32,2}) 
        mat = Float32(1.0 / (size(noise)[1] - 1)) * Hermitian(adjoint(noise)*noise)
        cholesky(mat).U |> inv
    end

    whitening_matrix = whitening_transform(noise_acq.data)
    noise_dwell_time = noise_acq.header.sample_time_us

    scaling_factor(acq::MRD.Acquisition) =
        sqrt(2 * acq.header.sample_time_us * noise_bandwidth / noise_dwell_time)

    apply_whitening(acq::MRD.Acquisition) = MRD.Acquisition(
        acq.header,
        scaling_factor(acq) * acq.data * whitening_matrix,
        acq.trajectory,
    )

    return Map(apply_whitening; spawn=true, buffer_size=Inf)
end


function remove_oversampling(header::MRDHeader)
    encoding_space = header.encoding[1].encodedSpace.matrixSize
    recon_space = header.encoding[1].reconSpace.matrixSize

    if encoding_space == recon_space
        return Map(identity)
    end


    x0 = (encoding_space.x - recon_space.x) ÷ 2 + 1
    x1 = x0 + recon_space.x -1 

    function crop_acquisition(acq::MRD.Acquisition)
        img_space = cifft(acq.data)             
        img_space = img_space[x0:x1, :]
        acq = @set acq.header.center_sample = recon_space.x ÷ 2
        acq.data = cfft(img_space)
        return acq
    end

    return Map(crop_acquisition; spawn=true, buffer_size=Inf)

end

function make_kspace(header::MRDHeader)

	matrix_size = header.encoding[1].encodedSpace.matrixSize
	
	function inner(acqs)
		number_of_samples, number_of_channels = size(acqs[1].data)
		buffer = zeros(ComplexF32,number_of_samples,matrix_size.y,matrix_size.z,number_of_channels)
		for acq in acqs 
			buffer[:,acq.header.idx.kspace_encode_step_1+1,acq.header.idx.kspace_encode_step_2+1,:] = acq.data
		end
		return acqs[1].header,buffer
		
	end

	return Map(inner)
end

function reconstruct_image(header::MRDHeader)

    field_of_view = header.encoding[1].reconSpace.fieldOfView_mm
    fov = (field_of_view.x,field_of_view.y,field_of_view.z)

	recon(x) = cifft(x,[1,2,3]) 
    coil_combine(x) = sqrt.(sum(abs2,x,dims=4))

	inner(index, reference::AcquisitionHeader, buffer) =   Image(ImageHeader(reference; image_index=index, image_type = MRD.ImageType.magnitude, field_of_view=fov),  coil_combine(recon(buffer)))
    inner_splat(args ) = inner(args[1],args[2][1],args[2][2])
	return Map(inner_splat)
end




function reconstruct_acquisitions(connection)
    start = time()
    header = connection.header
    noise_data = connection |>
       TakeWhile(acq -> :is_noise_measurement ∈ acq.header.flags) |> collect |> last 
    noise_whitener = noise_adjustment(header, noise_data)



    connection |>  noise_whitener |>
    remove_oversampling(header) |>
    SplitBy(acq -> :last_in_slice ∈ acq.header.flags, keepend = true) |>
    make_kspace(header) |>
    Enumerate() |>
    reconstruct_image(header) .|>
    push! $ connection

    tot_time = time()-start
    @info "Total recon time $tot_time"
end
