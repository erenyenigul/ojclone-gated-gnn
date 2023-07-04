from compy.datasets import Dataset
import tqdm, os
from tqdm import tqdm

class OJCloneDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

        self.additional_include_dirs = []
        self.content_dir = "source/OJClone"

    def preprocess(self, builder, visitor):
        suite_name = "OJClone"

        to_process = {}
        
        for label, folder_name in enumerate(os.listdir(self.content_dir)[:4]):
            f = os.path.join(self.content_dir, folder_name)
            if os.path.isfile(f): continue

            subfolder_name = os.path.join(self.content_dir, folder_name)
            
            for file_name in os.listdir(subfolder_name)[:20]:
                f = os.path.join(subfolder_name, file_name)
                if not os.path.isfile(f): continue

                if not file_name.endswith('.cpp'): continue

                file = open(f, "r")
                source_code = file.read()

                to_process[(
                    f,
                    source_code,
                    label,
                )] = []

        # Process the map of files
        processed = {}
        for file_data in tqdm(to_process, desc="Source Code -> IR+"):
            (
                #bench_file,
                file_name,
                source_code,
                #additional_include_dir,
                #suite_name,
                label,
                #benchmark_name,
            ) = file_data
         
            extractionInfo = builder.string_to_info(
                source_code
            )
            
            to_process[(file_name, source_code, label)] = extractionInfo
            
            

        # Map to dataset and extract representations
        samples = {}
        for file_data, function_datas in tqdm(
            to_process.items(), desc="IR+ -> ML Representation"
        ):
            (
                #bench_file,
                file_name,
                source_code,
                #additional_include_dir,
                #suite_name,
                #benchmark_name,
                label
            ) = file_data
            
            samples[
                file_data + ("1", "1", label)
            ] = builder.info_to_representation(function_datas, visitor)

        print("Size of dataset:", len(samples))
        print("Number of unique tokens:", builder.num_tokens())
        builder.print_tokens()
        print(builder.info_to_representation(function_datas, visitor))
        
        return {
            "samples": [
                {
                    "info": info,
                    "x": {"code_rep": sample, "aux_in": [0,0]},
                    "y": info[5],
                }
                for info, sample in samples.items()
            ],
            "num_types": builder.num_tokens(),
        }