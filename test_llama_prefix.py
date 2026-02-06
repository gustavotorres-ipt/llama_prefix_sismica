import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.oneformer.modeling_oneformer import PredictionBlock
from models.llama_prefix import LlamaPrefix
from models.clip_model import CLIP_DistilBert_ResNet
from config import CUSTOM_CLIP_FILE, device, LANG_PREFIX_CHECKPOINT
from dataset import load_datasets

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = False
import json
import os

DICT_EVENTS = {
    'sigmoid': [
        'a sigmoid', 'a clinoform', 'progradational geometry',
        'a sigmoidal clinoform set', 'an oblique clinoform pattern',
        'forward-prograding geometry',
    ],
    'shingled': [
        'overlapping geometry', 'superposed structure', 'shingled reflection',
        'imbricated reflection sets', 'shingled stratification',
        'layer-on-layer reflection geometry',
    ],
    'subparallel': [
        'nearly parallel reflection set', 'almost parallel reflectors',
        'weakly dipping layers', 'gently converging reflections',
        'subhorizontal bedding pattern', 'mildly inclined strata',
        'semi-parallel internal geometry', 'uniformly stratified reflections',
        'near-parallel lamination',
    ],
    'parallel': [
        'a parallel geometry', 'continuous horizontal reflections', 'a planar geometry',
        'an uniform layering pattern', 'a parallel-bedded strata',
        'concordant seismic layering', 'a tabular reflection pattern',
        'laterally uniform bedding', 'a parallel-laminated unit',
        'an horizontally layered sequence',
    ],
    'divergent': [
        'a divergent bedform geometry', 'a spreading stratigraphic unit',
        'diverging seismic pattern', 'flaring reflection geometry',
        'a fanning reflection geometry', 'thickening wedge pattern',
        'divergent foreset reflections', 'differentially dipping reflectors',
    ],
    'mounded': [
        'mounded geometry', 'domed structure', 'rounded pattern',
        'positive-relief geometry', 'convex-up reflection geometry',
        'mounded depositional body', 'positive-relief feature',
        'lenticular reflection pattern', 'dome-shaped seismic pattern',
    ],
    'deformed': [
        'disturbed reflection zone', 'folded reflector package',
        'distorted reflection set', 'structurally disturbed reflections',
        'a folded and faulted strata', 'deformation-related geometry',
        'a warped reflection package', 'deformed structure', 'distorted seismic face',
    ],
    'hummocky': [
        'a rugged reflector package', 'a corrugated seismic unit.',
        'a hummocky cross-stratified pattern', 'a low-relief mound-and-swale structure',
        'an uneven depositional surface', 'gently rolling internal reflections',
        'a corrugated seismic texture',
    ],
    'chaotic': [
        'chaotic reflections', 'an undefined seismic structure', 'a chaotic pattern',
        'undefined internal geometry', 'chaotic internal geometry',
        'internal structuraly disturbed reflexions',
    ],

    'chaotic-channels': [
        'disordered reflection patterns indicative of potential meandering channels',
        'irregular seismic reflections suggestive of channel systems',
        'undefined internal geometry with possibility of containing channels',
        'complex and chaotic reflection patterns indicating possible presence of channels',
        'chaotic seismic reflections potentially indicative of meandering channel systems',
    ],

    'wavy': [
        'a wavy structure', 'wavy seismic reflection geometry', 'wave-like seismic unit',
        'undulating bedding configuration', 'undulatory reflector pattern',
        'imbricated reflection sets', 'layer-on-layer reflection geometry',
    ],

}


def avaliar_legendas(legendas_geradas_labels):
    resultados = []
    total_corretos = 0

    for resultado_avaliado in legendas_geradas_labels:
        contagens = {}
        correto = False

        quantidade_captions = len(resultado_avaliado['captions'])

        for n in range(quantidade_captions):
            legenda_avaliada = resultado_avaliado['captions'][n]

            # checa se alguma das frases está na legenda avaliada
            for chave, lista_frases in DICT_EVENTS.items():
                for frase in lista_frases:
                    if frase in legenda_avaliada:
                        if chave in contagens:
                            contagens[chave] += 1
                        else:
                            contagens[chave] = 1

        if contagens:  # Verifica se encontrou algo
            label_maior_contagem = max(contagens, key=contagens.get)
            valor_contagem = contagens[label_maior_contagem]

            # Verificar se acertou
            if label_maior_contagem == resultado_avaliado['label']:
                correto = True
                total_corretos += 1
        else:
            label_maior_contagem = "Nenhuma frase encontrada"
            valor_contagem = 0

        resultados.append({
            #'arquivo': nome_arquivo,
            'maior_contagem': label_maior_contagem,
            'label': resultado_avaliado['label'],
            'contagem': valor_contagem,
            'acertou': correto
        })

        #se quiser visualizar por arquivo:
        #print(f"\nArquivo: {nome_arquivo}")
        print(f"Maior Contagem: {label_maior_contagem} ({valor_contagem})")
        print(f"Label: {resultado_avaliado['label']}")
        print(f"Correto: {correto}")
        print(f"Legenda avaliada: {legenda_avaliada}")
        print('----------------------')


    acuracia = total_corretos/len(resultados)*100

    print(f"\n Acurácia: {acuracia}")


def cut_text_after_last_period(text: str) -> str:
    idx = text.rfind(".")
    if idx == -1:
        return text
    return text[:idx] + '.'


def get_most_probable_tokens_prefix(llama_model, prefix_embeds):

    if prefix_embeds.dim() == 2:
        prefix_embeds = prefix_embeds.unsqueeze(0)

    token_embeds = llama_model.lang_model.model.embed_tokens.weight
    token_embeds = token_embeds.to(device)

    prefix_norm = F.normalize(prefix_embeds, dim=-1)
    token_norm  = F.normalize(token_embeds, dim=-1)

    sims = torch.einsum("bph,vh->bpv", prefix_norm, token_norm)

    top1_sims, top1_ids = torch.topk(sims, k=1, dim=-1)
    top1_sims = top1_sims.tolist()

    # Decode
    tokens = llama_model.tokenizer.convert_ids_to_tokens( top1_ids[0, :, 0].tolist())

    return tokens, top1_sims


def main():
    _, val_dataset = load_datasets()

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    legendas_geradas_labels = []

    with torch.no_grad():
        clip_encoder = CLIP_DistilBert_ResNet().to(device)
        clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
        clip_encoder.eval()

        llama_model = LlamaPrefix().to(device)
        llama_model.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))
        llama_model.eval()


        for image_batch, caption_batch, label_batch in tqdm(val_loader):
            image_batch = image_batch.to(device)
            image_embeds = clip_encoder.encode_image(image_batch)

            prefix_embeds = \
                llama_model.get_prefix_embeds_from_img_embeds(image_embeds)[0]
            prefix_embeds = prefix_embeds.unsqueeze(0)

            bos_id = llama_model.tokenizer.bos_token_id
            bos_embed = llama_model.lang_model.model.embed_tokens(
                torch.tensor([[bos_id]], device=device)
            )
            inputs_embeds = torch.cat([bos_embed, prefix_embeds], dim=1)

            print(100 * '-')
            #prefix_tokens, _ = get_most_probable_tokens_prefix(
            #    llama_model, inputs_embeds)

            print("Correct caption:", caption_batch[0])
            generated_text = llama_model.generate_text_from_embeds(inputs_embeds)
            generated_text = cut_text_after_last_period(generated_text)

            print("Generated:", generated_text)
            print("Label:", label_batch[0])
            #print("Closest tokens to prefix embeddings:", " ".join(prefix_tokens[1:]))


            legendas_geradas_labels.append({
                'captions': [generated_text],
                'label': label_batch[0]
            })
        avaliar_legendas(legendas_geradas_labels)


if __name__ == "__main__":
    main()
