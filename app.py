import streamlit as st
import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Recall, Precision

# --- CONSTANTES GLOBALES ---
# Liste des classes ASL (A-Z, del, nothing, space)
ASL_CLASSES = [chr(ord('A') + i) for i in range(26)]
ASL_CLASSES.extend(['del', 'nothing', 'space'])
NUM_CLASSES_ASL = len(ASL_CLASSES)

# Valeurs de normalisation pour ImageNet (standard pour les modèles pré-entraînés)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# --- Définition de WNBModule ---
# Copiez exactement votre classe WNBModule ici
class WNBModule(pl.LightningModule):
    def __init__(self, num_classes: int = NUM_CLASSES_ASL, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
             return self.model(x)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def unfreeze_all_layers(self):
         pass


# --- Nom du fichier State Dict (MODIFIEZ CE NOM LORS DU DÉPLOIEMENT) ---
# Ce nom de fichier DOIT CORRESPONDRE au nom du fichier .pth que vous avez téléchargé
# et placé dans le même répertoire que ce fichier app.py pour le déploiement.
MODEL_STATE_DICT_PATH = "asl_resnet_model.pt" # <--- VERIFIEZ CE NOM !


# --- Charger le modèle à partir du State Dict (Mis en cache par Streamlit) ---
@st.cache_resource
def load_model_state_dict(model_state_dict_path, num_classes):
    """Charge un modèle TorchScript sérialisé."""
    if not os.path.exists(model_state_dict_path):
        st.error(f"Erreur : Le fichier du modèle n'a pas été trouvé : {model_state_dict_path}")
        st.stop()
        return None

    st.write(f"Chargement du modèle TorchScript : {model_state_dict_path}")
    try:
        # Charger le modèle compilé TorchScript
        model = torch.jit.load(model_state_dict_path, map_location=torch.device('cpu'))

        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            st.write("Modèle chargé sur GPU.")
        else:
            model = model.cpu()
            st.write("Modèle chargé sur CPU.")

        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle TorchScript : {e}")
        st.stop()
        return None


# Charger le modèle au démarrage de l'application
model = load_model_state_dict(MODEL_STATE_DICT_PATH, NUM_CLASSES_ASL)


# --- Définir les transformations d'image ---
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Fonction pour faire une prédiction ---
def predict_on_image(image_pil):
    if model is None:
        return "Modèle non chargé.", "N/A"

    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        image_tensor = inference_transforms(image_pil).unsqueeze(0)

        # Déplacer le tensor image sur le même appareil que le modèle
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            image_tensor = image_tensor.cpu()


        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        conf, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = ASL_CLASSES[predicted_class_idx.item()]
        confidence = conf.item() * 100

        return predicted_class, f"{confidence:.2f}%"

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return "Erreur de prédiction", "N/A"


# --- Interface Utilisateur Streamlit ---
st.title("Détection du Langage des Signes Américain (ASL)")
st.write("Chargez une image d'une lettre ou d'un signe ASL (A-Z, del, nothing, space) pour obtenir la prédiction.")
st.write("Modèle : ResNet18 fine-tuné.")


uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée.", use_column_width=True)

    if st.button("Prédire"):
        st.write("")
        st.write("Classification en cours...")

        predicted_label, prediction_confidence = predict_on_image(image)

        st.write("--- Résultat ---")
        st.success(f"Classe prédite : **{predicted_label}**")
        st.info(f"Confiance : {prediction_confidence}")

st.sidebar.header("À propos")
st.sidebar.info(
    "Cette application utilise un modèle de réseau de neurones (ResNet18) entraîné "
    "pour reconnaître les signes du langage des signes américain (ASL)."
)
st.sidebar.write(f"Nombre de classes reconnues : {NUM_CLASSES_ASL}")
