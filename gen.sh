#!/bin/bash

# Standard-URL für das Inference-Script
INFERENCE_URL="https://raw.githubusercontent.com/webdebug/flux-train/main/inference.py"
WORKSPACE="/workspace"

# Laden des Inference-Scripts
echo "Lade das Inference-Script herunter..."
curl -o "$WORKSPACE/inference.py" -s "$INFERENCE_URL"
if [ $? -ne 0 ]; then
    echo "Fehler beim Herunterladen des Inference-Scripts."
    exit 1
fi
echo "Inference-Script gespeichert unter $WORKSPACE/inference.py."

# Installation der erforderlichen Python-Pakete
echo "Installiere erforderliche Python-Pakete..."
pip install "transformers[sentencepiece]" transformers accelerate peft diffusers safetensors gradio torch hf_transfer
if [ $? -ne 0 ]; then
    echo "Fehler bei der Installation der Pakete."
    exit 1
fi

# Alternative Torch-Version installieren (falls benötigt)
# echo "Installiere alternative Torch-Version..."
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Erstelle eine Token-Datei
TOKEN_FILE="$WORKSPACE/token"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "Erstelle Token-Datei..."
    touch "$TOKEN_FILE"
    echo "Token-Datei erstellt unter $TOKEN_FILE. Bitte füge deinen Hugging Face Token hinzu."
else
    echo "Token-Datei existiert bereits unter $TOKEN_FILE."
fi

echo "Setup abgeschlossen. Inference-Script und Umgebung bereit."
