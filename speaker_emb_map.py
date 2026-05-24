import gradio as gr
import numpy as np
import plotly.graph_objects as go
import umap
import torch
from reflow.vocoder import load_model_vocoder

# --- 1. Load Data ---
spk_map = {
    # "1": "Speaker A",
    # "2": "Speaker B",
    # etc...
}

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


# --- 2. Core Logic Functions ---

def generate_umap_map(n_neighbors, min_dist):
    """Runs UMAP and returns the new 2D coordinates dictionary and boundaries."""
    # Ensure n_neighbors isn't larger than our dataset
    n_neighbors = min(int(n_neighbors), len(valid_embeddings) - 1)
    if n_neighbors < 2: n_neighbors = 2

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        n_components=2,
        random_state=42 # Keeps the map from spinning randomly on refresh
    )
    embeds_2d = reducer.fit_transform(valid_embeddings)

    # Build the dictionary mapping IDs to new coordinates
    coords_dict = {}
    for i, spk_id in enumerate(valid_spk_ids):
        coords_dict[spk_id] = {
            "name": spk_map.get(spk_id, f"Voice {spk_id}"),
            "x": float(embeds_2d[i, 0]),
            "y": float(embeds_2d[i, 1])
        }

    return coords_dict, embeds_2d

def get_mix_from_coords(target_x, target_y, coords_dict, k_nearest=3):
    """Vectorized distance calculation to find the closest speakers."""
    if not coords_dict: return {}

    keys = list(coords_dict.keys())
    points = np.array([[d["x"], d["y"]] for d in coords_dict.values()])
    target = np.array([target_x, target_y])

    # Calculate distances
    distances = np.linalg.norm(points - target, axis=1)
    closest_idx = np.argsort(distances)[:k_nearest]

    # Inverse distance weighting
    weights = 1.0 / (distances[closest_idx] + 1e-6)
    weights /= weights.sum()

    return {keys[idx]: float(weight) for idx, weight in zip(closest_idx, weights)}

def draw_plot(x_val, y_val, coords_dict):
    """Draws the Plotly map using the current state."""
    fig = go.Figure()

    if not coords_dict:
        return fig

    # Extract data for a SINGLE Plotly trace (Massive performance boost)
    x_coords = [d["x"] for d in coords_dict.values()]
    y_coords = [d["y"] for d in coords_dict.values()]
    names = [d["name"] for d in coords_dict.values()]

    # Background Speakers
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        name='Speakers',
        hovertext=names, # Cleaner than text overlay
        marker=dict(size=8, color='cornflowerblue', opacity=0.7)
    ))

    # User's Custom Point
    fig.add_trace(go.Scatter(
        x=[x_val], y=[y_val],
        mode='markers',
        name='Custom Voice',
        marker=dict(size=18, color='red', symbol='star')
    ))

    # Calculate padding to lock axes
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_pad, y_pad = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2

    fig.update_layout(
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad], title="Latent Axis X"),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad], title="Latent Axis Y"),
        showlegend=False,
        title="Voice Latent Space Explorer",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# --- 3. Gradio Event Handlers ---

def update_explorer_xy(x_val, y_val, coords_dict):
    """Fires when the user moves the Latent X/Y sliders."""
    mix_dict = get_mix_from_coords(x_val, y_val, coords_dict)
    mix_text = ",\n".join([f"{coords_dict[k]['name']}: {v*100:.1f}%" for k, v in mix_dict.items()])
    fig = draw_plot(x_val, y_val, coords_dict)

    return fig, mix_text

def recalculate_umap(n_neighbors, min_dist):
    """Fires when the user changes the UMAP config."""
    # 1. Generate the new map
    coords_dict, embeds_2d = generate_umap_map(n_neighbors, min_dist)

    # 2. Find the new center of the map so the user's star doesn't get lost
    new_x = float(embeds_2d[:, 0].mean())
    new_y = float(embeds_2d[:, 1].mean())

    # 3. Calculate new slider boundaries
    x_min, x_max = embeds_2d[:, 0].min(), embeds_2d[:, 0].max()
    y_min, y_max = embeds_2d[:, 1].min(), embeds_2d[:, 1].max()
    x_pad, y_pad = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2

    # 4. Generate initial plot and text for the new map
    mix_dict = get_mix_from_coords(new_x, new_y, coords_dict)
    mix_text = ",\n".join([f"{coords_dict[k]['name']}: {v*100:.1f}%" for k, v in mix_dict.items()])
    fig = draw_plot(new_x, new_y, coords_dict)

    # 5. Return updates to ALL affected UI components
    return (
        fig,
        mix_text,
        gr.Slider(minimum=x_min - x_pad, maximum=x_max + x_pad, value=new_x), # Update X Slider
        gr.Slider(minimum=y_min - y_pad, maximum=y_max + y_pad, value=new_y), # Update Y Slider
        coords_dict # Update the hidden state
    )


# --- 4. Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Hidden state to store the current map coordinates across events
    map_state = gr.State({})

    gr.Markdown("# Speaker embeddings map")

    with gr.Row():
        with gr.Column(scale=1):

            with gr.Accordion("🗺️ Map Settings (UMAP)", open=False):
                gr.Markdown("Adjust how the voices are grouped. *Higher Neighbors = Traits, Lower Neighbors = Individuals.*")
                umap_neighbors = gr.Slider(minimum=2, maximum=50, value=15, step=1, label="N-Neighbors")
                umap_min_dist = gr.Slider(minimum=0.0, maximum=0.99, value=0.1, step=0.01, label="Min Distance")

            gr.Markdown("### Navigate Space")
            x_slider = gr.Slider(minimum=-10, maximum=10, value=0.0, step=0.01, label="X Axis")
            y_slider = gr.Slider(minimum=-10, maximum=10, value=0.0, step=0.01, label="Y Axis")

            gr.Markdown("### Current Recipe (`spk_mix_dict`)")
            mix_output = gr.Textbox(label="", lines=4)

        with gr.Column(scale=2):
            plot_output = gr.Plot()

    # Interactivity: When X/Y changes
    x_slider.change(update_explorer_xy, inputs=[x_slider, y_slider, map_state], outputs=[plot_output, mix_output])
    y_slider.change(update_explorer_xy, inputs=[x_slider, y_slider, map_state], outputs=[plot_output, mix_output])

    # Interactivity: When UMAP settings change
    # Use .release() instead of .change() so it doesn't recalculate 100 times while dragging
    umap_neighbors.release(
        recalculate_umap,
        inputs=[umap_neighbors, umap_min_dist],
        outputs=[plot_output, mix_output, x_slider, y_slider, map_state]
    )
    umap_min_dist.release(
        recalculate_umap,
        inputs=[umap_neighbors, umap_min_dist],
        outputs=[plot_output, mix_output, x_slider, y_slider, map_state]
    )

    # Initial Load
    demo.load(
        recalculate_umap,
        inputs=[umap_neighbors, umap_min_dist],
        outputs=[plot_output, mix_output, x_slider, y_slider, map_state]
    )

if __name__ == "__main__":
    demo.launch()