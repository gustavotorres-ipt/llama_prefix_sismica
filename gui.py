import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QFont
from caption_generator import CaptionGenerator


#class ImageNorm(object):
#    def __call__(self, x):
#        return (x - x.mean()) / (x.std() + 1e-6)


#class CaptionGenerator:
#    def __init__(self):
        # Load img transformation pipeline
#        self.encoder = CLIP_DistilBert_ResNet() 
#        self.decoder = LlamaPrefix()
#
#        self.encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
#        self.decoder.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))
#
#        self.transformation = transforms.Compose([
#            transforms.Resize((IMG_SIZE, IMG_SIZE)),
#            transforms.ToTensor(),
#            ImageNorm(),
#        ])
#
#    def generate_caption(self, input_image: str):
#        # Load image
#        image = Image.open(input_image).convert("RGB")
#
#        # Transform image
#        tensor_image = self.transformation(image)
#        tensor_image = tensor_image.unsqueeze(0)
#
        # Encode image
#        image_embeds = self.encoder.encode_image(tensor_image)
#
#        # Get preffix embeddings
#        prefix_embeds = \
#            self.decoder.get_prefix_embeds_from_img_embeds(image_embeds)[0]
#        
#        prefix_embeds = prefix_embeds.unsqueeze(0).to(device)
#        bos_id = self.decoder.tokenizer.bos_token_id
#        bos_embed = self.decoder.lang_model.model.embed_tokens(
#            torch.tensor([[bos_id]], device=device)
#        )
#        inputs_embeds = torch.cat([bos_embed, prefix_embeds], dim=1)
#
#        # Generate caption from image embeds
#        caption = self.decoder.generate_text_from_embeds(inputs_embeds)
#
#        print(caption)
#        if caption[-1] != '.':
#            caption = '.'.join(caption.split('.')[:-1]) + '.'
#
#        return caption

class ImageCaptionApp(QWidget):
    def __init__(self, caption_generator):
        super().__init__()

        self.caption_generator = caption_generator

        self.setWindowTitle("Exibição de Imagem Sísmica com Legenda")
        self.setGeometry(200, 200, 600, 400)
        font = QFont("Arial", 14)

        # Widgets
        self.image_label = QLabel("Nenhuma imagem carregada")
        self.image_label.setScaledContents(True)
        self.image_label.setFont(font)

        self.caption_label = QLabel("Legenda aparecerá aqui")
        self.caption_label.setWordWrap(True)

        self.caption_label.setFont(font)

        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.setFont(font)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

        # Conexão
        self.load_button.clicked.connect(self.load_image)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 300))

            legenda = self.lock_and_generate_caption(file_name)

            self.caption_label.setText(legenda)
            self.load_button.setDisabled(False)
            self.repaint()

    def lock_and_generate_caption(self, file_name):
        # Disable load button and change text to loading image...
        self.load_button.setDisabled(True)
        self.caption_label.setText("Gerando legenda...")
        self.repaint()

        # Generate the caption automatically
        caption = self.caption_generator.generate_caption(file_name)
        return caption


if __name__ == "__main__":
    app = QApplication(sys.argv)
    caption_generator = CaptionGenerator()
    window = ImageCaptionApp(caption_generator)
    window.show()
    sys.exit(app.exec_())
