# Standard library
import os

# Plotly for interactive visualizations
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Matplotlib for static visualizations
import matplotlib.pyplot as plt



def analyze(image, mask=None):
    """with plotly for detailed analysis"""
    if mask is None:
        fig = px.imshow(image)
        fig.show()
    fig = make_subplots(
    rows=1, cols=2)
    fig.add_trace(go.Image(z=image), 1, 1)
    fig.add_trace(go.Image(z=mask), 1, 2)
    fig.show()

def visualize(image, mask=None, masktitle='Mask'):
    plt.figure(figsize=(10, 5))
    
    if mask is None:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        im = plt.imshow(mask, cmap='jet', vmin=0, vmax=3)  # Set fixed colorbar range
        plt.title(masktitle)
        plt.axis('off')

        # Add colorbar (acts as legend)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Mask Intensity')

        plt.show()