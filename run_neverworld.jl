using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.ImmersedBoundaries: PartialCellBottom, GridFittedBottom

# using ClimaOcean.IdealizedSimulations: neverworld_simulation
using ClimaOcean.VerticalGrids: stretched_vertical_faces, PowerLawStretching
using Printf
using CUDA

include("neverworld_simulation.jl")

closure = CATKEVerticalDiffusivity(minimum_turbulent_kinetic_energy = 1e-6,
                                   minimum_convective_buoyancy_flux = 1e-11)

# z = stretched_vertical_faces(surface_layer_Δz = 8,
#                              surface_layer_height = 32,
#                              stretching = PowerLawStretching(1.02),
#                              maximum_Δz = 400.0,
#                              depth = 4000)

z = stretched_vertical_faces(surface_layer_Δz = 8,
                             surface_layer_height = 40,
                             stretching = PowerLawStretching(1.015),
                             maximum_Δz = 300.0,
                             depth = 4000)

ρ₀ = 1026 # kg m⁻³

latitudes      = [-80,   -45,   -15,      0,    15,    45,  80]
zonal_stresses = [ +0,  -0.2,  +0.1,  +0.02,  +0.1,  -0.1,  +0] ./ ρ₀ # kinematic wind stress
zonal_wind_stress = CubicSplineFunction{:y}(latitudes, zonal_stresses)

simulation = neverworld_simulation(CPU(); z,
                                   ImmersedBoundaryType = GridFittedBottom,
                                   horizontal_resolution = 1/4,
                                   longitude = (0, 50),
                                   latitude = (-80, 80),
                                   time_step = 5minutes,
                                   stop_time = 4 * 360days,
                                   closure)

model = simulation.model
grid = model.grid

using GLMakie

λ, φ, z = nodes(grid, Center(), Center(), Center())

fig = Figure(size = (800, 800))

ax_b = Axis(fig[1, 1],
            aspect = DataAspect(),
            xlabel = "longitude [ᵒ]",
            ylabel = "latitude [ᵒ]",
            xticks = -180:10:180,
            yticks = -90:10:90)

hm = heatmap!(ax_b, λ, φ, interior(grid.immersed_boundary.bottom_height, :, :, 1))

Colorbar(fig[1, 2], hm, label = "depth [m]")

ax_τˣ = Axis(fig[1, 3],
             xlabel = "zonal wind stress [N m⁻²]",
             ylabel = "latitude [ᵒ]",
             limits = (nothing, (-80, 80)),
             yticks = -90:10:90)

τˣ = - ρ₀ * model.velocities.u.boundary_conditions.top.condition[1, :]

lines!(ax_τˣ, τˣ, φ)

save("bathtub.png", fig)

current_figure()


fig = Figure(size = (300, 800))
ax = Axis(fig[1, 1], ylabel = "Depth (m)", xlabel = "Vertical spacing (m)")

lines!(ax, zspacings(grid.underlying_grid, Center()), znodes(grid, Center()))

scatter!(ax, zspacings(grid.underlying_grid, Center()), znodes(grid, Center()))

save("vertical_grid.png", fig)

current_figure()
