"""
fmri_encoder.py — Text → fMRI embedding encoder
=================================================

WHAT THIS SCRIPT DOES
---------------------
Downloads and runs a text encoder that produces brain-activity-style embeddings
from text input. Given a sentence or prompt, it returns a vector representing
predicted fMRI activation patterns in response to that text.

HOW IT WORKS
------------
The pipeline has two stages:

  Stage 1 — Semantic encoder (CLIP)
      CLIP's text encoder maps text into a 512-dimensional semantic space.
      This is not arbitrary: CLIP embeddings have been empirically shown to
      correlate strongly with fMRI responses in visual and language cortex
      (see: Scotti et al. 2023, Takagi et al. 2023, Allen et al. 2022).
      The model was trained on 400M image-text pairs, giving it rich
      semantic representations that align with human neural responses.

  Stage 2 — fMRI projection head
      A linear layer maps the 512-dim CLIP embedding into a higher-dimensional
      "voxel space" — the dimensionality of a typical fMRI ROI (region of
      interest). Default is 4096 dimensions, matching common ROI sizes in
      datasets like the Natural Scenes Dataset (NSD).

      By default this projection is random (untrained). To get subject-specific
      brain-aligned embeddings, you would fine-tune this head on paired
      (text, fMRI scan) data from a real subject — see FINE-TUNING section below.

WHY CLIP SPECIFICALLY
---------------------
CLIP was trained with a contrastive objective that aligns visual and language
representations. This alignment happens to mirror how the human brain integrates
multimodal information:
  - CLIP text embeddings predict fMRI responses in early visual cortex
    (V1–V4) and higher-level regions (EBA, FFA, PPA)
  - The NSD benchmark uses CLIP features as a standard brain encoding baseline
  - MindEye (Scotti et al. 2023) and Brain-Diffuser (Ozcelik et al. 2023)
    both use CLIP as the core alignment space for fMRI reconstruction

FINE-TUNING FOR REAL fMRI ALIGNMENT
-------------------------------------
If you have paired (text, fMRI) data, fine-tune the projection head:

    encoder = FMRITextEncoder()
    optimizer = torch.optim.AdamW(encoder.projection.parameters(), lr=1e-4)

    for text, fmri_scan in paired_data:
        pred = encoder.encode(text)
        loss = F.mse_loss(pred, fmri_scan)
        loss.backward()
        optimizer.step()

Public datasets with paired text/image + fMRI:
  - Natural Scenes Dataset (NSD): 8 subjects, 73k images with text captions
    https://naturalscenesdataset.org
  - The Little Prince fMRI: 51 subjects listening to audiobook
    https://openneuro.org/datasets/ds003643

USAGE
-----
  # Encode a single prompt
  python fmri_encoder.py --text "a red apple on a wooden table"

  # Encode multiple prompts
  python fmri_encoder.py --texts "a dog" "a car" "the ocean at night"

  # Encode from a file (one prompt per line) and save embeddings
  python fmri_encoder.py --file prompts.txt --output embeddings.npy

  # Change output dimensionality (default 4096 = NSD ROI size)
  python fmri_encoder.py --text "gravity" --voxels 1000
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Default voxel dimensionality — 4096 matches the size of a typical
# high-level ROI in the Natural Scenes Dataset (NSD).
DEFAULT_VOXELS = 4096

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FMRITextEncoder(nn.Module):
    """
    Text → fMRI embedding model.

    Architecture:
        text → [CLIP text encoder] → 512-dim semantic embedding
                                   → [linear projection]
                                   → N-dim fMRI voxel embedding

    The CLIP encoder is frozen. Only the projection head is trainable,
    making this efficient to fine-tune on small fMRI datasets.

    Args:
        voxel_dim:    Output dimensionality (number of voxels to predict).
                      4096 matches a typical high-level ROI in NSD.
        clip_model_id: HuggingFace model ID for the CLIP encoder.
    """

    def __init__(self, voxel_dim: int = DEFAULT_VOXELS, clip_model_id: str = CLIP_MODEL_ID):
        super().__init__()

        print(f"Downloading CLIP text encoder: {clip_model_id}...")
        self.tokenizer   = CLIPTokenizer.from_pretrained(clip_model_id)
        self.clip        = CLIPTextModel.from_pretrained(clip_model_id)
        self.clip_dim    = self.clip.config.hidden_size   # 512 for ViT-B/32
        self.voxel_dim   = voxel_dim

        # Freeze CLIP — only the projection head is trained
        for p in self.clip.parameters():
            p.requires_grad = False

        # Linear projection: semantic space → voxel space
        # In a real brain encoding model this would be fit to (text, fMRI) pairs.
        # Here it is initialised randomly as a usable default.
        self.projection = nn.Linear(self.clip_dim, voxel_dim)

        print(f"CLIP encoder ready  ({self.clip_dim}-dim → {voxel_dim}-dim voxel space)")

    def clip_embedding(self, text: str) -> torch.Tensor:
        """
        Run text through CLIP and return the pooled 512-dim embedding.

        CLIP uses the [EOS] token embedding as the sentence representation,
        which is then L2-normalised (as in the original CLIP paper).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,   # CLIP's maximum context length
        ).to(DEVICE)

        with torch.no_grad():
            outputs     = self.clip(**inputs)
            # Pooled output = [EOS] token embedding, L2-normalised
            pooled      = outputs.pooler_output            # (1, 512)
            normalised  = F.normalize(pooled, dim=-1)

        return normalised  # (1, 512)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into a predicted fMRI activation vector.

        Returns a 1D tensor of shape (voxel_dim,) representing predicted
        neural activation in response to the input text.
        """
        clip_emb    = self.clip_embedding(text)        # (1, 512)
        voxel_emb   = self.projection(clip_emb)        # (1, voxel_dim)
        return voxel_emb.squeeze(0)                    # (voxel_dim,)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of texts into a batch of fMRI embeddings.

        Returns a tensor of shape (N, voxel_dim).
        """
        embeddings = [self.encode(t) for t in texts]
        return torch.stack(embeddings)                 # (N, voxel_dim)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D embedding vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def print_embedding_summary(text: str, embedding: torch.Tensor):
    """Print a human-readable summary of an embedding."""
    arr = embedding.detach().cpu().numpy()
    print(f"\nText     : {text!r}")
    print(f"Shape    : {arr.shape}")
    print(f"Mean     : {arr.mean():.4f}")
    print(f"Std      : {arr.std():.4f}")
    print(f"Min/Max  : {arr.min():.4f} / {arr.max():.4f}")
    print(f"Preview  : [{', '.join(f'{v:.3f}' for v in arr[:6])}  ...]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode text into fMRI-style brain activity embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fmri_encoder.py --text "a red apple on a wooden table"
  python fmri_encoder.py --texts "a dog" "a car" "the ocean at night"
  python fmri_encoder.py --file prompts.txt --output embeddings.npy
  python fmri_encoder.py --text "gravity" --voxels 1000
        """,
    )
    parser.add_argument("--text",   type=str, default=None, help="Single text to encode")
    parser.add_argument("--texts",  type=str, nargs="+",    help="Multiple texts to encode")
    parser.add_argument("--file",   type=str, default=None, help="File with one text per line")
    parser.add_argument("--output", type=str, default=None, help="Save embeddings to .npy file")
    parser.add_argument("--voxels", type=int, default=DEFAULT_VOXELS,
                        help=f"Output voxel dimensionality (default: {DEFAULT_VOXELS})")
    args = parser.parse_args()

    # Collect texts
    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.texts:
        texts = args.texts
    elif args.text:
        texts = [args.text]
    else:
        # Default demo texts — chosen to span different semantic categories
        # so you can observe how the embeddings differ between them
        texts = [
            "a red apple on a wooden table",
            "the sound of rain on a roof at night",
            "a mathematical proof of Pythagoras' theorem",
            "the smell of coffee in the morning",
            "a crowded street in Tokyo",
        ]

    # Build encoder
    encoder = FMRITextEncoder(voxel_dim=args.voxels).to(DEVICE)
    encoder.eval()

    print(f"\nDevice  : {DEVICE}")
    print(f"Voxels  : {args.voxels}")
    print("=" * 60)

    # Encode and display
    all_embeddings = []
    for text in texts:
        emb = encoder.encode(text)
        print_embedding_summary(text, emb)
        all_embeddings.append(emb.detach().cpu().numpy())

    # Pairwise similarity if more than one text
    if len(texts) > 1:
        print("\n--- Pairwise cosine similarity ---")
        tensors = [encoder.encode(t) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = cosine_similarity(tensors[i], tensors[j])
                print(f"  {texts[i]!r:40s} ↔  {texts[j]!r:40s}  :  {sim:.3f}")

    # Save if requested
    if args.output:
        arr = np.stack(all_embeddings)
        np.save(args.output, arr)
        print(f"\nEmbeddings saved to {args.output}  shape={arr.shape}")
