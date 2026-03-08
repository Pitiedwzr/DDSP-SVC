import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from reflow.vocoder import load_model_vocoder

# 1. Configuration (Change to your actual model path!)
MODEL_PATH = "exp/film-test/model_100000.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Speaker JSON Map
# It will use the name if available, otherwise it falls back to the ID
try:
    with open("spk_info.json", "r", encoding="utf-8") as f:
        SPK_NAMES = json.load(f)
except FileNotFoundError:
    print("Warning: spk_info.json not found. Falling back to numeric IDs.")
    SPK_NAMES = {}

# 2. Load Model and Embeddings
print("Loading model and building Voice Map...")
model, vocoder, args = load_model_vocoder(MODEL_PATH, device=DEVICE)

# Extract Reflow's 256D speaker embeddings
embeddings = model.spk_embed.weight.detach().cpu().numpy()
num_speakers = embeddings.shape[0]

# 3. Crush down to 2D using PCA
pca = PCA(n_components=2)
points_2d = pca.fit_transform(embeddings)

# Create the visual map function
def plot_voice_map(clicked_point=None):
    fig, ax = plt.subplots(figsize=(10, 10)) # Made slightly larger for text readability
    
    # Plot all base speakers
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', alpha=0.5, s=40)
    
    # Add text labels (Names instead of IDs!)
    for i in range(num_speakers):
        str_id = str(i)
        # Get name from JSON, if not found, use "Spk {ID}"
        display_name = SPK_NAMES.get(str_id, f"Spk {str_id}")
        
        # Add a tiny offset to the text so it doesn't cover the dot
        ax.annotate(display_name, (points_2d[i, 0] + 0.02, points_2d[i, 1] + 0.02), 
                    fontsize=8, alpha=0.8, ha='left', va='bottom')
        
    # If the user clicked somewhere, show a red star there!
    if clicked_point is not None:
        ax.scatter(clicked_point[0], clicked_point[1], c='red', marker='*', s=300, label='Your Custom Voice')
        ax.legend()
        
    ax.set_title("Interactive Latent Voice Map")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Hide axis numbers since they are abstract PCA coordinates and don't mean anything to humans
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout so names on the edges don't get cut off
    plt.tight_layout()
    return fig

# 4. The Core Logic: Turn a click into a spk_mix_dict
def handle_map_click(evt: gr.SelectData):
    click_x, click_y = evt.index[0], evt.index[1]
    
    # Calculate distance from click to ALL speakers
    click_point = np.array([click_x, click_y])
    distances = np.linalg.norm(points_2d - click_point, axis=1)
    
    # Find the 3 closest speakers
    closest_idx = np.argsort(distances)[:3]
    closest_distances = distances[closest_idx]
    
    # Inverse Distance Weighting
    inv_dist = 1.0 / (closest_distances + 1e-6)
    weights = inv_dist / np.sum(inv_dist)
    
    # 1. Generate the RAW numeric dictionary for the model
    spk_mix_dict = {str(closest_idx[i]): float(weights[i]) for i in range(3)}
    
    # 2. Generate a HUMAN READABLE text box
    mix_text = "You created a mix of:\n\n"
    for k, v in spk_mix_dict.items():
        name = SPK_NAMES.get(k, f"Speaker {k}")
        # Format as a percentage (e.g., 65.2%)
        mix_text += f"🎤 {name} (ID: {k}) -> {v*100:.1f}%\n"
    
    # Format the raw dictionary string nicely
    raw_dict_str = "{\n"
    for k, v in spk_mix_dict.items():
        raw_dict_str += f"  '{k}': {v:.4f},\n"
    raw_dict_str += "}"
    
    # Update the plot to show the red star
    new_plot = plot_voice_map(clicked_point=(click_x, click_y))
    
    return new_plot, mix_text, raw_dict_str

# 5. Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ DDSP-Reflow Interactive Voice Map")
    gr.Markdown("Click anywhere on the map below! The system will mathematically blend the nearest speakers based on exactly where you click.")
    
    with gr.Row():
        with gr.Column(scale=5):
            # Show the plot
            map_plot = gr.Plot(value=plot_voice_map())
        
        with gr.Column(scale=2):
            gr.Markdown("### Your Custom Voice Profile")
            output_text = gr.Textbox(label="Mix Ratios", lines=5, interactive=False)
            output_dict = gr.Textbox(label="Raw spk_mix_dict (Copy & Paste to CLI)", lines=5, interactive=False)

    map_plot.select(
        handle_map_click, 
        inputs=None, 
        outputs=[map_plot, output_text, output_dict]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
