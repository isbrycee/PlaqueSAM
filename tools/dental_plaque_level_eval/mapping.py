from typing import Dict


class ImageToothMapping:
    def __init__(self):
        # Mapping of image to mouth
        img_to_mouth_mapping = {}
        with open('img_to_mouth_mapping.txt') as f:
            for line in f:
                img_id, mouth_id = line.strip().split(' ')
                img_to_mouth_mapping[int(img_id)] = int(mouth_id)

        # Tooth of interest for each mouth (Used for calculating plaque level accuracy)
        mouth_to_tooth_of_interests = {
            1: [51, 52, 61, 62, 11, 21],
            2: [71, 72, 81, 82, 31, 41],
            3: [53, 54, 55, 16],
            4: [83, 84, 85, 46],
            5: [63, 64, 65, 26],
            6: [73, 74, 75, 36]
        }

        # Mapping of the original tooth_id to new tooth_id
        class_name_to_idx_map = {
            '51': 0, '52': 1, '53': 2, '54': 3, '55': 4,
            '61': 5, '62': 6, '63': 7, '64': 8, '65': 9,
            '71': 10, '72': 11, '73': 12, '74': 13, '75': 14,
            '81': 15, '82': 16, '83': 17, '84': 18, '85': 19,

            '11': 20, '16': 21,
            '21': 22, '26': 23,
            '31': 24, '36': 25,
            '41': 26, '46': 27,

            'doubleteeth': 28,
            'crown': 29,

            '51_stain': 0, '52_stain': 1, '53_stain': 2, '54_stain': 3, '55_stain': 4,
            '61_stain': 5, '62_stain': 6, '63_stain': 7, '64_stain': 8, '65_stain': 9, '63_stan': 7,
            '71_stain': 10, '72_stain': 11, '73_stain': 12, '74_stain': 13, '75_stain': 14,
            '81_stain': 15, '82_stain': 16, '83_stain': 17, '84_stain': 18, '85_stain': 19,
            '71_stian': 10,

            '52_retainedteeth': 1,
            '53_retainedteeth': 2,
            '75_discoloration': 14,
            '51_discoloration': 0,
            '51_retainedteeth': 0,
            '61_retainedteeth': 5,
            '62_retainedteeth': 6,
            '64_retainedteeth': 8,
            '63_retainedteeth': 7,
            '54_retainedteeth': 3,
            '74_retainedteeth': 13,
            '61_discoloration': 5,

            '55_crown': 29,
            '84_crown': 29,
            '74_crown': 29,

            "55'": 4,
            '622': 6,

            # '585':19,
            # '875':14,

            '72\\3': 28,
            '72/3': 28,
            '82/83': 28,
            '81/82': 28,

            '110': 15,

            # '42':16,
            # '32':11,
            # '22': 0,
            # '23': 0,
            # '24': 0,
            # '25': 0,
        }

        self.img_toi_categories = {}
        for img_id, mouth_id in img_to_mouth_mapping.items():
            tooth_of_interests = mouth_to_tooth_of_interests[mouth_id]
            self.img_toi_categories[img_id] = {}
            for tooth in tooth_of_interests:
                tooth_id = class_name_to_idx_map[str(tooth)]
                # There are 3 category_id for each tooth (plaque, non-plaque, caries)
                self.img_toi_categories[img_id][tooth_id] = \
                    {tooth_id * 3: 'plaque', tooth_id * 3 + 1: 'non_plaque', tooth_id * 3 + 2: 'caries'}

    def __getitem__(self, img_id) -> Dict[int, Dict[int, str]]:
        return self.img_toi_categories[img_id]


if __name__ == '__main__':
    mapping = ImageToothMapping()
    print(mapping[1])
