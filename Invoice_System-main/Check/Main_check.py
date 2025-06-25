import os
import json
import torch
from PIL import Image
import easyocr
from transformers import BertTokenizerFast
from Check.LayoutLM import LayoutLMForTokenClassification
from Check.Evaluate import *
from Check.Preprocessing import is_image_file
from Check.Process_info import correct_text_label_pairs


address_label_order = [
    "address_do",
    "address_si", "address_gun", "address_gu",
    "address_eup", "address_myeon", "address_dong",
    "address_ri",
    "address_ro_name", "address_gil_name"
]

def load_processed_files(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return set(json.load(f))
    else:
        return set()

def save_processed_files(file_path, processed_files):
    with open(file_path, 'w') as f:
        json.dump(list(processed_files), f)

def Main_check():
    try:
        print("초기화 시작...")
        os.makedirs(r'.\Results', exist_ok=True)
        folder_path = r".\Preprocessing\rotated_and_filtered_invoice_5"
        font_path = r".\Font\NotoSansKR-Medium.ttf"
        processed_files_file = r".\processed_files.json"
        processed_files = load_processed_files(processed_files_file)

        print("모델 로딩...")
        reader = easyocr.Reader(['ko', 'en'], detector=True,
                                model_storage_directory=r'.\EasyOCR\workspace\user_network_dir',
                                user_network_directory=r'.\EasyOCR\workspace\user_network_dir',
                                recog_network='custom')

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = LayoutLMForTokenClassification()
        model.load_state_dict(torch.load(r".\trained_model.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print("데이터 로드 시작...")
        preprocessed_data = {}
        for file_name in os.listdir(folder_path):
            if file_name not in processed_files:
                processed_files.add(file_name)
                save_processed_files(processed_files_file, processed_files)
                file_path = os.path.join(folder_path, file_name)

                try:
                    image = Image.open(file_path).convert("RGB")
                    preprocessed_data[file_name] = {
                        'image': image,
                        'file_path': file_path
                    }
                    print(f"Successfully loaded {file_name}")
                except Exception as e:
                    print(f"Error loading {file_name}: {str(e)}")
                    continue

        for file_name, data in preprocessed_data.items():
            print(f"\nProcessing image: {file_name}")
            print("-" * 50)

            try:
                image = data['image']

                print("OCR 결과(GT) 불러오는 중...")
                gt_folder = './gt_easyocr'
                gt_path = os.path.join(gt_folder, os.path.splitext(file_name)[0] + ".json")
                with open(gt_path, encoding='utf-8') as f:
                    easyocr_results = json.load(f)

                print("\nLayoutLM 처리 중...")
                try:
                    layoutlm_inputs = process_image_for_layoutlm(
                        data['file_path'], tokenizer, device, easyocr_results
                    )

                    print("\nLayoutLM 입력 형태:")
                    for key, value in layoutlm_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"{key}: shape {value.shape}, dtype {value.dtype}")

                    with torch.no_grad():
                        layoutlm_outputs = model(
                            input_ids=layoutlm_inputs['input_ids'],
                            bbox=layoutlm_inputs['bbox'],
                            attention_mask=layoutlm_inputs['attention_mask'],
                            token_type_ids=layoutlm_inputs['token_type_ids'],
                            resized_images=layoutlm_inputs['resized_image'],
                            resized_and_aligned_bounding_boxes=layoutlm_inputs['resized_and_aligned_bounding_boxes']
                        )

                        layoutlm_results = process_layoutlm_outputs(
                            layoutlm_outputs,
                            tokenizer,
                            layoutlm_inputs['bbox'],
                            image,
                            easyocr_results
                        )

                        print(f"\nLayoutLM 결과: {len(layoutlm_results)} items found")

                    matched_pairs = []
                    for layout_item in layoutlm_results:
                        label = layout_item.get("label")
                        bbox = layout_item.get("bbox")
                        if not label or not bbox:
                            continue
                        for easy_item in easyocr_results:
                            if len(easy_item) < 2:
                                continue
                            easy_bbox = easy_item[0]
                            text = easy_item[1]
                            flat_easy_bbox = [coord for point in easy_bbox for coord in point]
                            min_x, min_y = min(flat_easy_bbox[::2]), min(flat_easy_bbox[1::2])
                            max_x, max_y = max(flat_easy_bbox[::2]), max(flat_easy_bbox[1::2])
                            if bbox == [min_x, min_y, max_x, max_y]:
                                matched_pairs.append((text, label))
                                break
                    
                    # 수신자 송신자 정보 담을 배열 선언
                    recipient_group = []
                    sender_group = []
                        
                    for text, label in matched_pairs:
                        if label.startswith("recipient_"):
                            recipient_group.append((text, label))
                        elif label.startswith("sender_"):
                            sender_group.append((text, label))
                    
                    # 각 그룹 정렬
                    recipient_group.sort(key=lambda x: address_label_order.index(x[1]) if x[1] in address_label_order else 999)
                    sender_group.sort(key=lambda x: address_label_order.index(x[1]) if x[1] in address_label_order else 999)
                        
                    print(f"text and label : {matched_pairs}")

                    corrected = correct_text_label_pairs(recipient_group + sender_group)
                    print("[CORRECTED RESULT]", ' '.join(corrected))

                    image.save(fr".\Results\{file_name}.jpg")

                except Exception as e:
                    print(f"LayoutLM 처리 중 오류 발생: {str(e)}")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

    except Exception as e:
        print(f"Main function error: {str(e)}")

    finally:
        print("처리 완료")


if __name__ == "__main__":
    Main_check()
