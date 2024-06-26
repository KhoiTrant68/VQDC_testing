import torch
from  modules.losses.vqperceptual import VQLPIPSWithDiscriminator

# Assuming `VQLPIPSWithDiscriminator` and other necessary components have been defined above.

# Dummy configurations for the discriminator and budget loss.
dummy_disc_config = {
    "target": "modules.discriminator.model.NLayerDiscriminator",
    "params": {
        "input_nc": 3,
        "ndf": 64,
        "n_layers": 3,
        "use_actnorm": False
    }
}

# dummy_budget_loss_config = {
#     "target": "modules.dynamic_modules.budget.BudgetConstraint_RatioMSE_DualGrain",
#     "params": {
#         "target_ratio": 0.5,
#         "gamma": 10.0,
#         "min_grain_size": 16,
#         "max_grain_size": 32,
#         "calculate_all": True
#     }
# }

dummy_budget_loss_config = None

# Instantiate the VQLPIPSWithDiscriminator model.
model = VQLPIPSWithDiscriminator(
    disc_start=1000,
    # disc_config=dummy_disc_config,
    # disc_init=True,
    codebook_weight=1.0,
    pixelloss_weight=1.0,
    disc_factor=1.0,
    disc_weight=1.0,
    perceptual_weight=1.0,
    disc_conditional=False,
    # disc_adaptive_loss=True,
    disc_loss="hinge",
    disc_weight_max=1.0,
    # budget_loss_config=dummy_budget_loss_config
)

# Create dummy input tensors.
batch_size = 4
input_channels = 3
image_size = 64
inputs = torch.randn(batch_size, input_channels, image_size, image_size)
reconstructions = torch.randn(batch_size, input_channels, image_size, image_size)
codebook_loss = torch.randn(batch_size)
global_step = 500
optimizer_idx = 0

# Perform a forward pass for generator update.
loss, log = model(
    codebook_loss=codebook_loss,
    inputs=inputs,
    reconstructions=reconstructions,
    optimizer_idx=optimizer_idx,
    global_step=global_step
)

print("Generator loss:", loss.item())
print("Log dictionary:", log)

# Perform a forward pass for discriminator update.
optimizer_idx = 1
d_loss, d_log = model(
    codebook_loss=codebook_loss,
    inputs=inputs,
    reconstructions=reconstructions,
    optimizer_idx=optimizer_idx,
    global_step=global_step
)

print("Discriminator loss:", d_loss.item())
print("Discriminator log dictionary:", d_log)
