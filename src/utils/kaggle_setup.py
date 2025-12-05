"""
Configuração do Kaggle API para download do dataset
"""

import os
import shutil
from pathlib import Path
from configs.config import COLAB_CONFIG


def setup_kaggle_api(kaggle_json_path=None):
    """
    Configura a API do Kaggle para download de datasets
    
    Args:
        kaggle_json_path (str): Caminho para o arquivo kaggle.json
    """
    
    if kaggle_json_path is None:
        kaggle_json_path = COLAB_CONFIG["kaggle_json_path"]
    
    kaggle_dir = Path(COLAB_CONFIG["kaggle_dir"])
    kaggle_dir.mkdir(exist_ok=True)
    
                                
    if os.path.exists(kaggle_json_path):
        shutil.copy2(kaggle_json_path, kaggle_dir / "kaggle.json")
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        print("✅ Kaggle API configurada com sucesso!")
        return True
    else:
        print(f"❌ Arquivo kaggle.json não encontrado em: {kaggle_json_path}")
        print("📋 Instruções:")
        print("1. Acesse https://www.kaggle.com/account")
        print("2. Clique em 'Create New API Token'")
        print("3. Baixe o arquivo kaggle.json")
        print("4. Faça upload para o Colab")
        return False


def download_dataset(dataset_name, output_dir=None):
    """
    Baixa um dataset do Kaggle
    
    Args:
        dataset_name (str): Nome do dataset no formato 'usuario/dataset'
        output_dir (str): Diretório de destino
    """
    
    if output_dir is None:
        output_dir = COLAB_CONFIG["data_dir"]
    
                                    
    Path(output_dir).mkdir(exist_ok=True)
    
                           
    cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir}"
    
    print(f"📥 Baixando dataset: {dataset_name}")
    print(f"📁 Destino: {output_dir}")
    
                      
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Dataset baixado com sucesso!")
        return True
    else:
        print("❌ Erro ao baixar dataset")
        return False


def extract_dataset(zip_path, extract_to=None):
    """
    Extrai arquivo zip do dataset
    
    Args:
        zip_path (str): Caminho para o arquivo zip
        extract_to (str): Diretório de extração
    """
    
    import zipfile
    
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"📦 Extraindo: {zip_path}")
    print(f"📁 Para: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extração concluída!")
        return True
    except Exception as e:
        print(f"❌ Erro na extração: {e}")
        return False


if __name__ == "__main__":
                    
    print("🔧 Configurando Kaggle API...")
    
    if setup_kaggle_api():
        print("📥 Baixando dataset Libras MNIST...")
        download_dataset("datamoon/libras-mnist")
        
                         
        zip_file = Path(COLAB_CONFIG["data_dir"]) / "libras-mnist.zip"
        if zip_file.exists():
            extract_dataset(str(zip_file))
    else:
        print("❌ Configure primeiro o arquivo kaggle.json")
