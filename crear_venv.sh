#!/bin/bash
# ============================================
# 🚀 Script rápido para crear y activar entorno virtual
# Autor: Facundo San Martino
# ============================================

set -e  # Detener si algún comando falla

echo "🧱 Instalando python3-venv..."
sudo apt install -y python3-venv

echo "✨ Creando entorno virtual..."
python3 -m venv venv

echo "📦 Activando entorno virtual..."
source venv/bin/activate

echo "✅ Entorno virtual creado y activado."
echo "👉 Si querés volver a activarlo más tarde: source venv/bin/activate"
