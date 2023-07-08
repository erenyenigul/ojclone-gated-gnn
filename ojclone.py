from compy.datasets import Dataset
import tqdm, os
from tqdm import tqdm

class OJCloneDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

        self.additional_include_dirs = []
        self.content_dir = "compiled/source/OJClone"
    
    def preprocess(self, builder, visitor):
        suite_name = "OJClone"

        to_process = {}

        for label, folder_name in enumerate(os.listdir(self.content_dir)):
            f = os.path.join(self.content_dir, folder_name)
            if os.path.isfile(f): continue

            subfolder_name = os.path.join(self.content_dir, folder_name)
            
            for file_name in os.listdir(subfolder_name):

                f = os.path.join(subfolder_name, file_name)
                if not os.path.isfile(f): continue

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
            
            # Sometimes builder causes segfault, we can
            # skip those programs.
            try:
                extractionInfo = builder.string_to_info(
                    source_code
                )
            
                to_process[(file_name, source_code, label)] = extractionInfo
            except Exception as e:
                print(e)
                pass
            
        # Map to dataset and extract representations
        samples = {}
        for file_data, function_datas in tqdm(to_process.items(), desc="IR+ -> ML Representation"):
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
                file_data
            ] = builder.info_to_representation(function_datas, visitor)

        print("Size of dataset:", len(samples))
        print("Number of unique tokens:", builder.num_tokens())
        builder.print_tokens()
        
        return {
            "samples": [
                {
                    "info": {"filename": info[0], "label": info[2]},
                    "x": {"code_rep": sample, "aux_in": [0,0]},
                    "y": info[2],
                }
                for info, sample in samples.items()
            ],
            "num_types": builder.num_tokens(),
        }