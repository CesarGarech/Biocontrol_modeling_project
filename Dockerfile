# 1. Usar una imagen oficial de Python ligera pero completa
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema operativo
# Incluye build-essential para librerías de C/C++ (necesario a veces por CasADi/SciPy)
# e incluye una distribución básica de TeX Live para la generación de reportes LaTeX/PDF
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    gnupg2 \
    wget \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# 4. Instalar .NET 8 runtime (requerido por DWSIM 9 en Linux)
#    Se agrega el repositorio de paquetes de Microsoft para Debian.
RUN . /etc/os-release \
    && wget -q "https://packages.microsoft.com/config/debian/${VERSION_ID}/packages-microsoft-prod.deb" \
           -O /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && rm /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y dotnet-runtime-8.0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Instalar DWSIM 9 para Linux (paquete .deb oficial para Debian/Ubuntu)
#    DWSIM provee paquetes .deb probados en Ubuntu 22 y Ubuntu 24.
#    Para actualizar: cambiar VERSION y el nombre del archivo al release más reciente.
#    Releases disponibles en: https://github.com/DanWBR/dwsim/releases
#    Formato de URL: https://github.com/DanWBR/dwsim/releases/download/v{VERSION}/dwsim_{VERSION}-amd64.deb
RUN wget -q "https://github.com/DanWBR/dwsim/releases/download/v9.0.5/dwsim_9.0.5-amd64.deb" \
        -O /tmp/dwsim.deb \
    && dpkg -i /tmp/dwsim.deb || true \
    && apt-get install -f -y \
    && rm -f /tmp/dwsim.deb \
    && rm -rf /var/lib/apt/lists/*

# 6. Variables de entorno para DWSIM
#    DWSIM_INSTALL_PATH: directorio donde el paquete .deb instala los binarios de DWSIM.
#    USE_DWSIM_LIVE: habilita la integración en vivo con DWSIM (en lugar del modo analítico).
ENV DWSIM_INSTALL_PATH=/usr/lib/dwsim
ENV USE_DWSIM_LIVE=true

# 7. Copiar primero el archivo de requerimientos
# Esto optimiza el caché de Docker. Si cambias el código pero no los requerimientos,
# Docker no volverá a descargar todas las librerías.
COPY requirements.txt .

# 8. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 9. Copiar el resto del código del repositorio al contenedor
COPY . .

# 10. Exponer el puerto de la aplicación (8501 por defecto para Streamlit)
# Si migras a Dash, cambia esto a 8050
EXPOSE 8501

# 11. Comando por defecto para ejecutar la aplicación web
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]