import torch
from transformers import StableDiffusionModel, StableDiffusionTokenizer

# Ladda in modellen och tokenisatorn
model_id = "CompVis/stablediffusion-v1-4"
model = StableDiffusionModel.from_pretrained(model_id)
tokenizer = StableDiffusionTokenizer.from_pretrained(model_id)

# Skapa en funktion för att generera bilder
def generate_image(prompt, width=512, height=512):
    # Tokenisera prompten
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generera bild
    output = model.generate(input_ids, num_steps=100)
    
    # Återvinna bild från tensor
    image = output[0].permute(1, 2, 0).numpy()
    
    # Rensa bild och spara som Bild
    from PIL import Image
    img = Image.fromarray(image)
    img.save("output.png")

# Använd funktionen för att generera en bild
generate_image("En vacker dag i Paris", width=512, height=512)
