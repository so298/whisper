import whisper
import torch

model = whisper.load_model("base")

encoder = torch.jit.script(model.encoder).eval().half().cpu()
decoder = torch.jit.script(model.decoder).eval().half().cpu()
# print(encoder)

encoder.save("torchscript-models/base-encoder.pt")
decoder.save("torchscript-models/base-decoder.pt")

# model.encoder = encoder
# model.decoder = decoder

# checkpoint = {
#     "encoder": encoder,
#     "decoder": decoder
# }
# torch.save(checkpoint, "checkpoint.pt")