# 1. Usar una imagen oficial de Python ligera pero completa
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema operativo
# Incluye build-essential para librerías de C/C++ (necesario a veces por CasADi/SciPy)
# e incluye una distribución básica de TeX Live para la generación de reportes LaTeX/PDF
RUN apt-get update && apt-get install -y \
    build-essential \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar primero el archivo de requerimientos
# Esto optimiza el caché de Docker. Si cambias el código pero no los requerimientos, 
# Docker no volverá a descargar todas las librerías.
COPY requirements.txt .

# 5. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar el resto del código del repositorio al contenedor
COPY . .

# 7. Exponer el puerto de la aplicación (8501 por defecto para Streamlit)
# Si migras a Dash, cambia esto a 8050
EXPOSE 8501

# 8. Comando por defecto para ejecutar la aplicación web
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]