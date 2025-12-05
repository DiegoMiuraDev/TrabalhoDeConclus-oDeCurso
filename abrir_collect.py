#!/usr/bin/env python3
"""Script auxiliar para abrir o coletor de dados"""
import subprocess
import sys
import os

print("="*60)
print("ABRINDO COLETOR DE DADOS DE TESTE")
print("="*60)
print()

script_path = os.path.join("scripts", "collect_test_data_debug.py")

if not os.path.exists(script_path):
    print(f"❌ Erro: Arquivo não encontrado: {script_path}")
    sys.exit(1)

print(f"✅ Arquivo encontrado: {script_path}")
print("🚀 Executando...")
print()
print("="*60)
print("INSTRUÇÕES:")
print("  - Uma janela da câmera deve abrir")
print("  - Pressione 1-5 para selecionar classe (A, E, I, O, U)")
print("  - Pressione ESPAÇO para começar/parar coleta")
print("  - Pressione 'q' para sair")
print("="*60)
print()

try:
    # Executar o script
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print(f"✅ Processo iniciado (PID: {process.pid})")
    print("📹 Verifique se a janela da câmera abriu")
    print()
    print("Pressione Ctrl+C para encerrar este script auxiliar")
    print("(Isso não fechará a janela da câmera)")
    print()
    
    # Aguardar processo
    stdout, stderr = process.communicate()
    
    if stdout:
        print("SAÍDA:")
        print(stdout)
    
    if stderr:
        print("ERROS:")
        print(stderr)
    
    print(f"\nProcesso finalizado com código: {process.returncode}")
    
except KeyboardInterrupt:
    print("\n⚠️  Interrompido pelo usuário")
    if process.poll() is None:
        print("Processo ainda está rodando...")
except Exception as e:
    print(f"\n❌ Erro: {e}")
    import traceback
    traceback.print_exc()




