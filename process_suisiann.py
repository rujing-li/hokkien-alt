# https://i3thuan5.github.io/tai5-uan5_gian5-gi2_kang1-ku7
from datasets import load_dataset
from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音

# Function to convert Tailo to number tones
def convert_tailo_format(tailo_text):
    try:
        tailo_object = 拆文分析器.建立句物件(tailo_text)
        return tailo_object.轉音(臺灣閩南語羅馬字拼音).看語句()
    except Exception as e:
        print(f"Error converting Tailo text: {e}")
        return ""

# Load the SuíSiann dataset
dataset_path = 'data/suisiann/SuiSiann.csv'  # Replace with the actual path to your SuíSiann dataset
dataset = load_dataset('csv', data_files={'full': dataset_path})['full']

# Process the dataset to extract and convert the necessary columns
def process_suisiann_row(example):
    try:
        hok_audio = example['音檔']  # Map '音檔' to 'hok_audio'
        hok_text_tailo_number_tone = convert_tailo_format(example['羅馬字'])
        return {'hok_audio': hok_audio, 'hok_text_tailo_number_tone': hok_text_tailo_number_tone}
    except KeyError as e:
        print(f"Missing column in row: {e}")
        return None

# Apply the processing function to the dataset
processed_dataset = dataset.map(process_suisiann_row, remove_columns=dataset.column_names)

# Filter out any None rows (in case of processing errors)
processed_dataset = processed_dataset.filter(lambda x: x is not None)

# Save the processed dataset to a new CSV file
processed_dataset.to_csv('data/suisiann/all.csv', index=False)  # Replace with desired output path

print("Processing complete. Saved to 'data/suisiann/all.csv'")
