#!/bin/bash
# Criar o ambiente virtual se ainda não existir
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Ativar o ambiente virtual
source venv/bin/activate

# Instalar Gunicorn se não estiver instalado
pip install gunicorn

# Iniciar o servidor Flask usando Gunicorn
exec gunicorn --bind 0.0.0.0:5000 app:app
