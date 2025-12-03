#!/bin/bash
# ============================================
# 🚀 Script rápido para crear y activar entorno virtual
# ============================================

set -e  # Detener si algún comando falla

echo "🧱 Actualizando repositorios..."
sudo apt update -y

# Detectar versión de Python instalada (por ejemplo: 3.10)
PY_VER=$(python3 -V 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PACKAGE="python${PY_VER}-venv"

echo "🐍 Detectada versión de Python: $PY_VER"
echo "📦 Instalando paquete: $PACKAGE"

# Instalar el paquete correspondiente (manejar casos donde no exista)
if sudo apt install -y "$PACKAGE"; then
    echo "✅ Paquete $PACKAGE instalado correctamente."
else
    echo "⚠️ No se encontró $PACKAGE, intentando con python3-venv genérico..."
    sudo apt install -y python3-venv || {
        echo "❌ Error: No se pudo instalar python3-venv. Verificá tus repositorios."
        exit 1
    }
fi

# Crear el entorno virtual
echo "✨ Creando entorno virtual..."
python3 -m venv venv

# Activar el entorno virtual
echo "📦 Activando entorno virtual..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "✅ Entorno virtual creado y activado."
echo "👉 Para volver a activarlo más tarde, ejecutá:"
echo "   source venv/bin/activate"
