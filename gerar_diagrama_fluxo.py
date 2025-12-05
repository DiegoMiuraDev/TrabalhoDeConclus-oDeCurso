#!/usr/bin/env python3
"""
Script para gerar diagrama de fluxo do sistema de interpretação de Libras
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Dimensões da imagem (aumentadas para não cortar)
WIDTH = 2000
HEIGHT = 900
BG_COLOR = (255, 255, 255)

# Cores - apenas preto e branco
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (240, 240, 240)

# Criar imagem
img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Tentar carregar fonte, senão usar padrão
try:
    title_font = ImageFont.truetype("arial.ttf", 28)
    box_font = ImageFont.truetype("arial.ttf", 16)
    arrow_font = ImageFont.truetype("arial.ttf", 14)
except:
    title_font = ImageFont.load_default()
    box_font = ImageFont.load_default()
    arrow_font = ImageFont.load_default()

# Título
title = "Arquitetura e Fluxo de Processamento do Sistema de Interpretação Automática de Libras"
title_bbox = draw.textbbox((0, 0), title, font=title_font)
title_width = title_bbox[2] - title_bbox[0]
title_x = (WIDTH - title_width) // 2
draw.text((title_x, 30), title, fill=BLACK, font=title_font)

# Configurações dos blocos
BOX_WIDTH = 220
BOX_HEIGHT = 140
BOX_SPACING = 50
START_X = 120
START_Y = 180
ARROW_LENGTH = 100

# Lista de blocos - todos em preto e branco
blocks = [
    {
        "title": "1. Captura de Vídeo",
        "text": "Câmera / OpenCV\nCaptura contínua\nde quadros",
        "color": LIGHT_GRAY
    },
    {
        "title": "2. Detecção de Mão",
        "text": "MediaPipe\nDetecção e recorte\nda ROI da mão",
        "color": LIGHT_GRAY
    },
    {
        "title": "3. Pré-processamento",
        "text": "Redimensionamento\nNormalização\nPreparação do input",
        "color": LIGHT_GRAY
    },
    {
        "title": "4. Classificação IA",
        "text": "Modelo Keras/\nTensorFlow Lite\nInferência da classe",
        "color": LIGHT_GRAY
    },
    {
        "title": "5. Suavização",
        "text": "Filtragem temporal\nEstabilização de\npredições",
        "color": LIGHT_GRAY
    },
    {
        "title": "6. Interface Gráfica",
        "text": "Tkinter\nExibição da letra,\nconfiança e vídeo",
        "color": LIGHT_GRAY
    }
]

# Desenhar blocos e setas
x = START_X
y = START_Y

for i, block in enumerate(blocks):
    # Desenhar retângulo do bloco
    box_x1 = x
    box_y1 = y
    box_x2 = x + BOX_WIDTH
    box_y2 = y + BOX_HEIGHT
    
    # Sombra (cinza claro)
    shadow_offset = 3
    draw.rectangle(
        [box_x1 + shadow_offset, box_y1 + shadow_offset, 
         box_x2 + shadow_offset, box_y2 + shadow_offset],
        fill=GRAY
    )
    
    # Bloco principal (cinza claro com borda preta)
    draw.rectangle(
        [box_x1, box_y1, box_x2, box_y2],
        fill=block["color"],
        outline=BLACK,
        width=3
    )
    
    # Título do bloco (preto)
    title_lines = block["title"].split("\n")
    title_y = box_y1 + 15
    for line in title_lines:
        text_bbox = draw.textbbox((0, 0), line, font=box_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = box_x1 + (BOX_WIDTH - text_width) // 2
        draw.text((text_x, title_y), line, fill=BLACK, font=box_font)
        title_y += 22
    
    # Texto do bloco (preto)
    text_lines = block["text"].split("\n")
    text_y = box_y1 + 60
    for line in text_lines:
        text_bbox = draw.textbbox((0, 0), line, font=box_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = box_x1 + (BOX_WIDTH - text_width) // 2
        draw.text((text_x, text_y), line, fill=BLACK, font=box_font)
        text_y += 20
    
    # Desenhar seta para o próximo bloco (exceto no último)
    if i < len(blocks) - 1:
        arrow_start_x = box_x2
        arrow_start_y = y + BOX_HEIGHT // 2
        arrow_end_x = x + BOX_WIDTH + ARROW_LENGTH
        arrow_end_y = arrow_start_y
        
        # Linha da seta (mais grossa)
        draw.line(
            [(arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y)],
            fill=BLACK,
            width=4
        )
        
        # Cabeça da seta (triângulo maior)
        arrow_head_size = 12
        arrow_points = [
            (arrow_end_x, arrow_end_y),
            (arrow_end_x - arrow_head_size, arrow_end_y - arrow_head_size),
            (arrow_end_x - arrow_head_size, arrow_end_y + arrow_head_size)
        ]
        draw.polygon(arrow_points, fill=BLACK, outline=BLACK)
        
        # Texto da seta (opcional)
        if i == 0:
            arrow_text = "Fluxo de quadros"
            text_bbox = draw.textbbox((0, 0), arrow_text, font=arrow_font)
            text_y = arrow_start_y - 25
            text_x = arrow_start_x + (ARROW_LENGTH - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, text_y), arrow_text, fill=BLACK, font=arrow_font)
        elif i == 1:
            arrow_text = "ROI da mão"
            text_bbox = draw.textbbox((0, 0), arrow_text, font=arrow_font)
            text_y = arrow_start_y - 25
            text_x = arrow_start_x + (ARROW_LENGTH - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, text_y), arrow_text, fill=BLACK, font=arrow_font)
        elif i == 2:
            arrow_text = "Imagem processada"
            text_bbox = draw.textbbox((0, 0), arrow_text, font=arrow_font)
            text_y = arrow_start_y - 25
            text_x = arrow_start_x + (ARROW_LENGTH - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, text_y), arrow_text, fill=BLACK, font=arrow_font)
        elif i == 3:
            arrow_text = "Probabilidades"
            text_bbox = draw.textbbox((0, 0), arrow_text, font=arrow_font)
            text_y = arrow_start_y - 25
            text_x = arrow_start_x + (ARROW_LENGTH - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, text_y), arrow_text, fill=BLACK, font=arrow_font)
        elif i == 4:
            arrow_text = "Letra estabilizada"
            text_bbox = draw.textbbox((0, 0), arrow_text, font=arrow_font)
            text_y = arrow_start_y - 25
            text_x = arrow_start_x + (ARROW_LENGTH - (text_bbox[2] - text_bbox[0])) // 2
            draw.text((text_x, text_y), arrow_text, fill=BLACK, font=arrow_font)
    
    # Atualizar posição para próximo bloco
    x += BOX_WIDTH + ARROW_LENGTH

# Adicionar legenda na parte inferior
legend_y = HEIGHT - 80
legend_text = "Sistema opera em tempo real com laço contínuo de captura, detecção, classificação e atualização da interface"
legend_bbox = draw.textbbox((0, 0), legend_text, font=arrow_font)
legend_width = legend_bbox[2] - legend_bbox[0]
legend_x = (WIDTH - legend_width) // 2
draw.text((legend_x, legend_y), legend_text, fill=BLACK, font=arrow_font)

# Salvar imagem
output_path = "diagrama_fluxo_sistema.png"
img.save(output_path)
print(f"✅ Diagrama gerado com sucesso: {output_path}")

