#!/usr/bin/env python3
"""
Setup do Ollama + tunel ngrok.
Originalmente eram 5 celulas de notebook, agora unificadas em um unico arquivo.
Basta rodar: python setup_ollama_ngrok.py
"""

import os
import time
import subprocess


def instalar_dependencias():
    # Instala as ferramentas necessarias (hardware + Ollama + biblioteca do ngrok)
    subprocess.run(["apt-get", "install", "-y", "zstd", "lshw", "pciutils", "-q"])
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    subprocess.run(["pip", "install", "pyngrok", "-q"])


def verificar_gpu():
    # Confirma se a GPU esta disponivel
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


def subir_ollama_e_baixar_modelo():
    # Inicia o servidor Ollama, baixa o modelo e verifica se carregou
    os.system("OLLAMA_HOST=0.0.0.0 CUDA_VISIBLE_DEVICES=0 ollama serve > /tmp/ollama.log 2>&1 &")
    time.sleep(10)

    subprocess.run(["ollama", "pull", "qwen2.5:14b"], timeout=600)
    print("Modelo pronto!")
    print(subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], capture_output=True, text=True).stdout)


def abrir_tunel_ngrok():
    # Cria a configuracao, expoe o Ollama na internet e mostra a GPU
    os.makedirs("/root/.config/ngrok", exist_ok=True)
    config = """version: "3"
agent:
  authtoken: 3EKvtOo5nbGjVkAFxXMIbNjQx5z_5M53aUu4ox5y7StVVpqHC
tunnels:
  ollama:
    proto: http
    addr: 11434
    domain: semicolon-wasp-user.ngrok-free.dev
    schemes:
      - https
"""
    with open("/root/.config/ngrok/ngrok.yml", "w") as f:
        f.write(config)

    subprocess.Popen(["ngrok", "start", "ollama", "--log", "/tmp/ngrok.log"])
    time.sleep(5)
    print("Servidor online em: https://semicolon-wasp-user.ngrok-free.dev")
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


def reiniciar_ollama():
    # (Opcional) Mata a instancia antiga, sobe o servidor novamente e testa a conexao
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(3)

    os.system("OLLAMA_HOST=0.0.0.0 CUDA_VISIBLE_DEVICES=0 ollama serve > /tmp/ollama.log 2>&1 &")
    time.sleep(8)
    print(subprocess.run(["curl", "-s", "http://localhost:11434"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    instalar_dependencias()
    verificar_gpu()
    subir_ollama_e_baixar_modelo()
    abrir_tunel_ngrok()
    # reiniciar_ollama()  # descomente esta linha se precisar reiniciar o servidor