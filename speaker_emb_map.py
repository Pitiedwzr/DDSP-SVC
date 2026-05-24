import gradio as gr
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch
from reflow.vocoder import load_model_vocoder

# Load Speaker Dictionary
spk_map = {
}

# Load the Model
model_path = "exp/film-low-lr/model_100000.pt"

print("Loading model and extracting embeddings...")
model, vocoder, args = load_model_vocoder(model_path, device='cpu')

all_embeddings = model.spk_embed.weight.detach().cpu().numpy() # shape: (n_spk, 256)

# Filter to only the speakers present in the dictionary
valid_spk_ids = []
valid_embeddings = []

for spk_id_str, spk_name in spk_map.items():
    idx = int(spk_id_str) - 1
    if idx < all_embeddings.shape[0]:
        valid_spk_ids.append(spk_id_str)
        valid_embeddings.append(all_embeddings[idx])
    else:
        print(f"Warning: ID {spk_id_str} is out of bounds for the embedding layer.")

valid_embeddings = np.array(valid_embeddings)

# PCA Dimensionality Reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(valid_embeddings)

# Map 2D coordinates back to the speaker ID
spk_2d_coords = {}
for i, spk_id_str in enumerate(valid_spk_ids):
    spk_2d_coords[spk_id_str] = {
        "name": spk_map[spk_id_str],
        "x": float(embeddings_2d[i, 0]),
        "y": float(embeddings_2d[i, 1])
    }

# Calculate dynamic boundaries for the UI sliders (with 20% padding)
x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
x_pad, y_pad = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2

# Math: Coordinates -> spk_mix_dict
def get_mix_from_coords(target_x, target_y, k_nearest=3):
    distances = {}
    for spk_id, data in spk_2d_coords.items():
        dist = np.sqrt((target_x - data["x"])**2 + (target_y - data["y"])**2)
        distances[spk_id] = dist

    # Get closest speakers
    closest = sorted(distances.items(), key=lambda item: item[1])[:k_nearest]

    # Inverse distance weighting (closer = higher weight)
    weights = np.array([1.0 / (dist + 1e-6) for _, dist in closest])
    weights /= weights.sum() # Normalize to 1.0

    mix_dict = {closest[i][0]: float(weights[i]) for i in range(len(closest))}
    return mix_dict

# UI Update Function
def update_explorer(x_val, y_val):
    # Calculate the mix dictionary
    mix_dict = get_mix_from_coords(x_val, y_val)

    # Format text for display
    mix_text = ",\n".join([f"{spk_2d_coords[k]['name']} (ID {k}): {v*100:.1f}%" for k, v in mix_dict.items()])

    # Draw Plotly graph
    fig = go.Figure()

    # Plot existing speakers
    for spk_id, data in spk_2d_coords.items():
        fig.add_trace(go.Scatter(
            x=[data["x"]], y=[data["y"]],
            mode='markers+text',
            name=data["name"],
            text=[data["name"]], textposition="top center",
            marker=dict(size=8, color='cornflowerblue', opacity=0.7)
        ))

    # Plot the User's Custom Point
    fig.add_trace(go.Scatter(
        x=[x_val], y=[y_val],
        mode='markers',
        name='Custom Voice',
        marker=dict(size=18, color='red', symbol='star')
    ))

    # Lock the axes so the graph doesn't jump
    fig.update_layout(
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad], title="Latent Axis X"),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad], title="Latent Axis Y"),
        showlegend=False,
        title="Voice Latent Space Explorer",
        height=600
    )

    # Generate audio placeholder
    # mel = model(units, f0, volume, spk_mix_dict=mix_dict, ...)
    # audio = vocoder.infer(mel, f0)
    # return fig, mix_text, audio

    return fig, mix_text

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 AI Voice Constellation")
    gr.Markdown("Move the sliders to navigate the latent space. The model will automatically blend the 3 closest speakers to create your new voice.")

    with gr.Row():
        with gr.Column(scale=1):
            x_slider = gr.Slider(minimum=x_min - x_pad, maximum=x_max + x_pad, value=0.0, step=0.01, label="X Axis (Timbre Trait A)")
            y_slider = gr.Slider(minimum=y_min - y_pad, maximum=y_max + y_pad, value=0.0, step=0.01, label="Y Axis (Timbre Trait B)")

            gr.Markdown("### Current Recipe (`spk_mix_dict`)")
            mix_output = gr.Textbox(label="", lines=4)

            # Placeholder for generation
            # generate_btn = gr.Button("Generate Audio", variant="primary")
            # audio_output = gr.Audio(label="Generated Audio")

        with gr.Column(scale=2):
            plot_output = gr.Plot()

    # Interactivity
    x_slider.change(update_explorer, inputs=[x_slider, y_slider], outputs=[plot_output, mix_output])
    y_slider.change(update_explorer, inputs=[x_slider, y_slider], outputs=[plot_output, mix_output])

    demo.load(update_explorer, inputs=[x_slider, y_slider], outputs=[plot_output, mix_output])

if __name__ == "__main__":
    demo.launch()