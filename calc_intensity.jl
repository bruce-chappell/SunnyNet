using Unitful
using HDF5
using Mmap
using FortranFiles
using Interpolations
import PhysicalConstants.CODATA2018: h, k_B, R_∞, c_0, m_e, m_u, e, ε_0, a_0
using Transparency
using ProgressMeter
using Base.Threads
@derived_dimension NumberDensity Unitful.𝐋^-3
@derived_dimension PerLength Unitful.𝐋^-1

const Hα = AtomicLine(97489.992u"cm^-1", 82257.172u"cm^-1", 109754.578u"cm^-1",
                    18, 8, 6.411e-01, 1.008 * m_u, 1)
const γ_energy = (h * c_0 / (4 * π * Hα.λ0)) |> u"J"
const unsold_const = const_unsold(Hα)
const quad_stark_const = const_quadratic_stark(Hα)


"""
Reads atmosphere in the input format of MULTI3D. Returns a dictionary.
"""
function read_atmos_multi3d(par_file, atmos_file; dtype=Float32)
    data = Dict()
    # Get parameters and height scale
    fobj = FortranFile(par_file, "r")
    _ = read(fobj, Int32)
    data[:nx] = read(fobj, Int32)
    data[:ny] = read(fobj, Int32)
    data[:nz] = read(fobj, Int32)
    _ = read(fobj, (Float64, data[:nx]))
    _ = read(fobj, (Float64, data[:ny]))
    data[:z] = read(fobj, (Float64, data[:nz]))u"cm"
    close(fobj)
    # Get atmosphere
    fobj = open(atmos_file, "r")
    shape = (data[:nx], data[:ny], data[:nz])
    block_size = data[:nx] * data[:ny] * data[:nz] * sizeof(dtype)
    data[:ne] = Mmap.mmap(fobj, Array{typeof(one(dtype)u"cm^-3"), 3}, shape)
    data[:temperature] = Mmap.mmap(fobj, Array{typeof(one(dtype)u"K"), 3}, shape, block_size)
    data[:vz] = Mmap.mmap(fobj, Array{typeof(one(dtype)u"cm/s"), 3}, shape, block_size * 4)
    close(fobj)
    return data
end


"""
Reads NLTE populations from MULTI3D output
"""
function read_multi3d_pops(pop_file, shape; dtype=Float32)
    fobj = open(pop_file, "r")
    pops = Mmap.mmap(fobj, Array{typeof(one(dtype)u"cm^-3"), 4}, shape) # NLTE pops
    close(fobj)
    return pops
end


"""
Reads NLTE population predictions from SunnyNet. The input file is an HDF5 file
written by SunnyNet.sunnynet_predict_populations().
"""
function read_sunnynet_pops(pop_file)
    pop_nn = h5read(pop_file, "populations")
    tmp = h5readattr(pop_file, "populations/")
    cmass_mean = tmp["cmass_mean"]
    cmass_scale = tmp["cmass_scale"]
    nz = length(cmass_mean)
    pop_nn = permutedims(pop_nn, [2, 1, 3, 4])
    nhydr, ny, nx = size(pop_nn)[end-2:end]
    new_pop = zeros(Float64, (nx, ny, nz, nhydr))
    p = ProgressMeter.Progress(nx * ny * nhydr, desc="Reading populations ")
    Threads.@threads for ij in CartesianIndices((nx, ny, nhydr))
        i, j, l = Tuple(ij)
        interp = LinearInterpolation(cmass_scale, pop_nn[:, l, j, i], extrapolation_bc=Line())
        new_pop[i, j, :, l] .= interp(cmass_mean) * 1.0f-6
        ProgressMeter.next!(p)
    end
    return new_pop * u"cm^-3"
end


"""
Compute continuum extinction.
"""
function αcont(λ::Unitful.Length, temperature::Unitful.Temperature,
               electron_density::NumberDensity, h_ground_density::NumberDensity,
               proton_density::NumberDensity)
    α = Transparency.hminus_ff_stilley(λ, temperature, h_ground_density, electron_density)
    α += Transparency.hminus_bf_geltman(λ, temperature, h_ground_density, electron_density)
    α += hydrogenic_ff(c_0 / λ, temperature, electron_density, proton_density, 1)
    α += h2plus_ff(λ, temperature, h_ground_density, proton_density)
    α += h2plus_bf(λ, temperature, h_ground_density, proton_density)
    α += thomson(electron_density)
    α += rayleigh_h(λ, h_ground_density)
    return α
end


"""
Calculates full spectrum for a given 1D column
"""
function calc_Halpha_1D(
        z_scale::Array{<: Unitful.Length, 1},
        wave::Array{<: Unitful.Length, 1},
        temp::Array{<: Unitful.Temperature, 1},
        ne::Array{<: NumberDensity, 1},
        v_los::Array{<: Unitful.Velocity, 1},
        hpops::Array{<: NumberDensity, 2},
)
    nwave = length(wave)
    nz = length(z_scale)
    intensity = zeros(nwave)u"kW / (m^2 * nm)"
    α_total = zeros(nz)u"m^-1"
    source_function = zeros(nz)u"kW / (m^2 * nm)"
    # Calculate continuum opacity and wavelength-independent quantities
    α_cont = αcont.(Hα.λ0, temp, ne, hpops[:, 1], hpops[:, 6])
    j_cont = α_cont .* (blackbody_λ.(Hα.λ0, temp) .|> u"kW / (m^2 * nm)")
    γ = γ_unsold.(unsold_const, temp, hpops[:, 1]) # van der Waals broadening
    γ .+= 5.701E+08u"s^-1" # from the atom file, not Hα.Aji  # natural broadening
    γ .+= γ_linear_stark.(ne, 3, 2)
    γ .+= γ_quadratic_stark.(ne, temp, stark_constant=quad_stark_const)
    ΔλD = doppler_width.(Hα.λ0, Hα.atom_weight, temp)
    # Calculate line opacity and intensity
    for (i, λ) in enumerate(wave)
        for iz in 1:nz
            # Wavelength-dependent part
            a = damping(γ[iz], λ, ΔλD[iz])
            v = (λ - Hα.λ0 + Hα.λ0 * v_los[iz] / c_0) / ΔλD[iz]
            profile = voigt_profile(a, v, ΔλD[iz])
            # Part that only multiplies by wavelength:
            α_tmp = γ_energy * profile
            j_tmp = α_tmp
            α_tmp *= hpops[iz, 2] * Hα.Bij - hpops[iz, 3] * Hα.Bji
            j_tmp *= hpops[iz, 3] * Hα.Aji
            α_tmp += α_cont[iz]
            j_tmp += j_cont[iz]
            source_function[iz] = j_tmp / α_tmp
            α_total[iz] = α_tmp
        end
        intensity[i] = piecewise_1D_linear(z_scale, α_total, source_function)[1]
    end
    return intensity
end


"""
Performs spectral synthesis for Hα, using NLTE populations from MULTI3D
and predicted NLTE populations from SunnyNet (for comparison). Also needs
an existing 3D model atmosphere in the input format of MULTI3D. File names
and paths can be edited.

FILE_WAVES is an hdf5 file with an array of wavelengths (nm in vacuum)
to compute the line profile.

This example reads the atmosphere from the output of MULTI3D, but this
can be changed. Change the lines of read_atmos_multi3d() to read your
simulation format. 'atmos' needs to be a dictionary with the following keys
for atmospheric arrays :z, :temperature, :ne, :vz (need to have units using
Unitful), plus the auxiliary dimensions :nx, :ny, and :nz.
"""
function do_calc()
    FILE_WAVES = "wavelengths.hdf5"
    M3D_DIR = "/path/to/multi3d/output/"
    SUNNYNET_POPS = "sunnynet_predicted.hdf5"
    OUTPUT = "sunnynet_intensity.hdf5"
    waves = h5read(FILE_WAVES, "wavelength")u"nm"
    nwave = length(waves)
    atmos = read_atmos_multi3d(joinpath(M3D_DIR, "out_par"), joinpath(M3D_DIR, "out_atm"))
    nx, ny, nz = atmos[:nx], atmos[:ny], atmos[:nz]
    lo, hi = (1, nz)  # use to slice the height scale, if needed
    z_scale = atmos[:z][lo:hi] .|> u"m"
    if isfile(OUTPUT)
        println("Intensity already calculated, refusing to overwrite.")
    else
        hpops = read_nn_pops(SUNNYNET_POPS)
        intensity = zeros(nwave, nx, ny)u"kW / (m^2 * nm)"
        p = ProgressMeter.Progress(nx * ny, desc="Calculating spectra ")
        Threads.@threads for ij in CartesianIndices((nx, ny))
            ix, iy = Tuple(ij)
            intensity[:, ix, iy] = calc_Halpha_1D(
                z_scale, waves, atmos[:temperature][ix, iy, lo:hi],
                atmos[:ne][ix, iy, lo:hi], atmos[:vz][ix, iy, lo:hi] .|> u"m/s",
                hpops[ix, iy, lo:hi, :]
            )
            ProgressMeter.next!(p)
        end
        h5write(OUTPUT, "intensity", ustrip(intensity))
    end
end


do_calc()
