#!/usr/bin/env python3
"""
Script para gerar um diagrama de alto nível (Interface -> Processos -> Resultados)
do sistema de interpretação automática de Libras.

A ideia é ter uma figura bem simples para o TCC, mostrando apenas:
    - Interface com o usuário
    - Processos internos
    - Resultados apresentados
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Dimensões da imagem
WIDTH = 1600
HEIGHT = 600
BG_COLOR = (255, 255, 255)

# Cores (preto e tons de cinza para boa impressão)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (240, 240, 240)
GRAY = (180, 180, 180)


def load_fonts():
    """Tenta carregar fontes TrueType; se não achar, usa a padrão."""
    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        box_title_font = ImageFont.truetype("arial.ttf", 22)
        box_text_font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        title_font = ImageFont.load_default()
        box_title_font = ImageFont.load_default()
        box_text_font = ImageFont.load_default()
    return title_font, box_title_font, box_text_font


def draw_centered_multiline(draw, box, text, font, fill=BLACK, line_spacing=6):
    """
    Desenha um texto multi-linha centralizado dentro de um retângulo.

    :param draw: ImageDraw.Draw
    :param box: (x1, y1, x2, y2)
    :param text: str com "\n" para quebras de linha
    """
    x1, y1, x2, y2 = box
    lines = text.split("\n")

    # Altura total do bloco de texto
    line_heights = []
    line_widths = []
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_widths.append(w)
        line_heights.append(h)
        total_height += h
    total_height += line_spacing * (len(lines) - 1 if len(lines) > 0 else 0)

    # Posição inicial (topo) para centralizar verticalmente
    current_y = y1 + (y2 - y1 - total_height) // 2

    for i, line in enumerate(lines):
        w = line_widths[i]
        h = line_heights[i]
        x = x1 + (x2 - x1 - w) // 2
        draw.text((x, current_y), line, font=font, fill=fill)
        current_y += h + line_spacing


def main():
    # Cria imagem
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_font, box_title_font, box_text_font = load_fonts()

    # Título da figura
    title = "Fluxo Geral do Sistema de Interpretação Automática de Libras"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]
    title_x = (WIDTH - title_w) // 2
    draw.text((title_x, 30), title, font=title_font, fill=BLACK)

    # Área útil para os três blocos
    margin_side = 80
    margin_top = 140
    margin_bottom = 80

    available_width = WIDTH - 2 * margin_side
    box_width = int(available_width / 3) - 20
    box_height = HEIGHT - margin_top - margin_bottom
    spacing_x = (available_width - 3 * box_width) // 2

    boxes = []
    for i in range(3):
        x1 = margin_side + i * (box_width + spacing_x)
        y1 = margin_top
        x2 = x1 + box_width
        y2 = y1 + box_height
        boxes.append((x1, y1, x2, y2))

    # Definição dos três blocos principais
    blocks = [
        {
            "title": "Interface\ndo Usuário",
            "text": (
                "Aplicativo desktop em Tkinter\n"
                "Seleção de câmera\n"
                "Botão para iniciar/parar captura\n"
                "Janela com vídeo em tempo real"
            ),
        },
        {
            "title": "Processos\nInternos",
            "text": (
                "Captura de quadros via OpenCV\n"
                "Detecção e recorte da mão (MediaPipe)\n"
                "Pré-processamento da imagem\n"
                "Classificação pelo modelo de IA\n"
                "Suavização das predições ao longo do tempo"
            ),
        },
        {
            "title": "Resultados\nApresentados",
            "text": (
                "Exibição da letra reconhecida\n"
                "Indicação do nível de confiança\n"
                "Atualização contínua do vídeo\n"
                "Geração de dados para análise e discussão"
            ),
        },
    ]

    # Desenha blocos + sombras
    shadow_offset = 4
    for i, (box, block) in enumerate(zip(boxes, blocks)):
        x1, y1, x2, y2 = box

        # Sombra
        draw.rectangle(
            (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset),
            fill=GRAY,
        )

        # Bloco
        draw.rectangle((x1, y1, x2, y2), fill=LIGHT_GRAY, outline=BLACK, width=3)

        # Linha horizontal para separar título do corpo
        title_height = 70
        draw.line((x1, y1 + title_height, x2, y1 + title_height), fill=BLACK, width=2)

        # Título
        draw_centered_multiline(
            draw,
            (x1, y1 + 10, x2, y1 + title_height - 5),
            block["title"],
            font=box_title_font,
            fill=BLACK,
        )

        # Texto
        draw_centered_multiline(
            draw,
            (x1 + 20, y1 + title_height + 10, x2 - 20, y2 - 10),
            block["text"],
            font=box_text_font,
            fill=BLACK,
        )

    # Desenha setas entre os blocos
    def draw_arrow_horizontal(x_start, y_center, x_end):
        # linha
        draw.line((x_start, y_center, x_end, y_center), fill=BLACK, width=4)
        # cabeça
        head_size = 12
        draw.polygon(
            [
                (x_end, y_center),
                (x_end - head_size, y_center - head_size),
                (x_end - head_size, y_center + head_size),
            ],
            fill=BLACK,
        )

    # Centro vertical das setas
    arrow_y = margin_top + box_height // 2

    # Entre Interface e Processos
    interface_box = boxes[0]
    processos_box = boxes[1]
    draw_arrow_horizontal(interface_box[2] + 10, arrow_y, processos_box[0] - 10)

    # Entre Processos e Resultados
    resultados_box = boxes[2]
    draw_arrow_horizontal(processos_box[2] + 10, arrow_y, resultados_box[0] - 10)

    # Legenda embaixo
    legend = (
        "Fluxo conceitual: o usuário interage com a interface, que aciona os processos internos "
        "de captura, detecção e classificação, culminando na apresentação dos resultados."
    )
    legend_bbox = draw.textbbox((0, 0), legend, font=box_text_font)
    legend_w = legend_bbox[2] - legend_bbox[0]
    legend_x = (WIDTH - legend_w) // 2
    legend_y = HEIGHT - margin_bottom + 20
    draw.text((legend_x, legend_y), legend, font=box_text_font, fill=BLACK)

    # Salvar imagem
    output_path = "diagrama_fluxo_alto_nivel.png"
    img.save(output_path)
    print(f"✅ Diagrama de alto nível gerado com sucesso: {output_path}")


if __name__ == "__main__":
    main()


