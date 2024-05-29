#!/bin/bash
# Navegar para o diretório backend
cd "$(dirname "$0")"

# Criar o ambiente virtual se ainda não existir
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Ativar o ambiente virtual
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar as dependências
pip install -r requirements.txt

# Executar o servidor Flask
exec python app.py
