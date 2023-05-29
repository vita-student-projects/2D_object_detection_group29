import numpy as np


COCO_BOX_SKELETON = [

    (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (5, 4),
         
]




KINEMATIC_TREE_SKELETON = [

    (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (5, 4),

]


COCO_KEYPOINTS = [
    'person_box_center', 'person_box_left_up_corner', 'person_box_right_up_corner', 'person_box_left_down_corner', 'person_box_right_down_corner',
    'bicycle_box_center', 'bicycle_box_left_up_corner', 'bicycle_box_right_up_corner', 'bicycle_box_left_down_corner', 'bicycle_box_right_down_corner',
    'car_box_center', 'car_box_left_up_corner', 'car_box_right_up_corner', 'car_box_left_down_corner', 'car_box_right_down_corner',
    'motorcycle_box_center', 'motorcycle_box_left_up_corner', 'motorcycle_box_right_up_corner', 'motorcycle_box_left_down_corner', 'motorcycle_box_right_down_corner',
    'airplane_box_center', 'airplane_box_left_up_corner', 'airplane_box_right_up_corner', 'airplane_box_left_down_corner', 'airplane_box_right_down_corner',
    'bus_box_center', 'bus_box_left_up_corner', 'bus_box_right_up_corner', 'bus_box_left_down_corner', 'bus_box_right_down_corner',
    'train_box_center', 'train_box_left_up_corner', 'train_box_right_up_corner', 'train_box_left_down_corner', 'train_box_right_down_corner',
    'truck_box_center', 'truck_box_left_up_corner', 'truck_box_right_up_corner', 'truck_box_left_down_corner', 'truck_box_right_down_corner',
    'boat_box_center', 'boat_box_left_up_corner', 'boat_box_right_up_corner', 'boat_box_left_down_corner', 'boat_box_right_down_corner',
    'traffic light_box_center', 'traffic light_box_left_up_corner', 'traffic light_box_right_up_corner', 'traffic light_box_left_down_corner', 'traffic light_box_right_down_corner',
    'fire hydrant_box_center', 'fire hydrant_box_left_up_corner', 'fire hydrant_box_right_up_corner', 'fire hydrant_box_left_down_corner', 'fire hydrant_box_right_down_corner',
    'stop sign_box_center', 'stop sign_box_left_up_corner', 'stop sign_box_right_up_corner', 'stop sign_box_left_down_corner', 'stop sign_box_right_down_corner',
    'parking meter_box_center', 'parking meter_box_left_up_corner', 'parking meter_box_right_up_corner', 'parking meter_box_left_down_corner', 'parking meter_box_right_down_corner',
    'bench_box_center', 'bench_box_left_up_corner', 'bench_box_right_up_corner', 'bench_box_left_down_corner', 'bench_box_right_down_corner',
    'bird_box_center', 'bird_box_left_up_corner', 'bird_box_right_up_corner', 'bird_box_left_down_corner', 'bird_box_right_down_corner',
    'cat_box_center', 'cat_box_left_up_corner', 'cat_box_right_up_corner', 'cat_box_left_down_corner', 'cat_box_right_down_corner',
    'dog_box_center', 'dog_box_left_up_corner', 'dog_box_right_up_corner', 'dog_box_left_down_corner', 'dog_box_right_down_corner',
    'horse_box_center', 'horse_box_left_up_corner', 'horse_box_right_up_corner', 'horse_box_left_down_corner', 'horse_box_right_down_corner',
    'sheep_box_center', 'sheep_box_left_up_corner', 'sheep_box_right_up_corner', 'sheep_box_left_down_corner', 'sheep_box_right_down_corner',
    'cow_box_center', 'cow_box_left_up_corner', 'cow_box_right_up_corner', 'cow_box_left_down_corner', 'cow_box_right_down_corner',
    'elephant_box_center', 'elephant_box_left_up_corner', 'elephant_box_right_up_corner', 'elephant_box_left_down_corner', 'elephant_box_right_down_corner',
    'bear_box_center', 'bear_box_left_up_corner', 'bear_box_right_up_corner', 'bear_box_left_down_corner', 'bear_box_right_down_corner',
    'zebra_box_center', 'zebra_box_left_up_corner', 'zebra_box_right_up_corner', 'zebra_box_left_down_corner', 'zebra_box_right_down_corner',
    'giraffe_box_center', 'giraffe_box_left_up_corner', 'giraffe_box_right_up_corner', 'giraffe_box_left_down_corner', 'giraffe_box_right_down_corner',
    'backpack_box_center', 'backpack_box_left_up_corner', 'backpack_box_right_up_corner', 'backpack_box_left_down_corner', 'backpack_box_right_down_corner',
    'umbrella_box_center', 'umbrella_box_left_up_corner', 'umbrella_box_right_up_corner', 'umbrella_box_left_down_corner', 'umbrella_box_right_down_corner',
    'handbag_box_center', 'handbag_box_left_up_corner', 'handbag_box_right_up_corner', 'handbag_box_left_down_corner', 'handbag_box_right_down_corner',
    'tie_box_center', 'tie_box_left_up_corner', 'tie_box_right_up_corner', 'tie_box_left_down_corner', 'tie_box_right_down_corner',
    'suitcase_box_center', 'suitcase_box_left_up_corner', 'suitcase_box_right_up_corner', 'suitcase_box_left_down_corner', 'suitcase_box_right_down_corner',
    'frisbee_box_center', 'frisbee_box_left_up_corner', 'frisbee_box_right_up_corner', 'frisbee_box_left_down_corner', 'frisbee_box_right_down_corner',
    'skis_box_center', 'skis_box_left_up_corner', 'skis_box_right_up_corner', 'skis_box_left_down_corner', 'skis_box_right_down_corner',
    'snowboard_box_center', 'snowboard_box_left_up_corner', 'snowboard_box_right_up_corner', 'snowboard_box_left_down_corner', 'snowboard_box_right_down_corner',
    'sports ball_box_center', 'sports ball_box_left_up_corner', 'sports ball_box_right_up_corner', 'sports ball_box_left_down_corner', 'sports ball_box_right_down_corner',
    'kite_box_center', 'kite_box_left_up_corner', 'kite_box_right_up_corner', 'kite_box_left_down_corner', 'kite_box_right_down_corner',
    'baseball bat_box_center', 'baseball bat_box_left_up_corner', 'baseball bat_box_right_up_corner', 'baseball bat_box_left_down_corner', 'baseball bat_box_right_down_corner',
    'baseball glove_box_center', 'baseball glove_box_left_up_corner', 'baseball glove_box_right_up_corner', 'baseball glove_box_left_down_corner', 'baseball glove_box_right_down_corner',
    'skateboard_box_center', 'skateboard_box_left_up_corner', 'skateboard_box_right_up_corner', 'skateboard_box_left_down_corner', 'skateboard_box_right_down_corner',
    'surfboard_box_center', 'surfboard_box_left_up_corner', 'surfboard_box_right_up_corner', 'surfboard_box_left_down_corner', 'surfboard_box_right_down_corner',
    'tennis racket_box_center', 'tennis racket_box_left_up_corner', 'tennis racket_box_right_up_corner', 'tennis racket_box_left_down_corner', 'tennis racket_box_right_down_corner',
    'bottle_box_center', 'bottle_box_left_up_corner', 'bottle_box_right_up_corner', 'bottle_box_left_down_corner', 'bottle_box_right_down_corner',
    'wine glass_box_center', 'wine glass_box_left_up_corner', 'wine glass_box_right_up_corner', 'wine glass_box_left_down_corner', 'wine glass_box_right_down_corner',
    'cup_box_center', 'cup_box_left_up_corner', 'cup_box_right_up_corner', 'cup_box_left_down_corner', 'cup_box_right_down_corner',
    'fork_box_center', 'fork_box_left_up_corner', 'fork_box_right_up_corner', 'fork_box_left_down_corner', 'fork_box_right_down_corner',
    'knife_box_center', 'knife_box_left_up_corner', 'knife_box_right_up_corner', 'knife_box_left_down_corner', 'knife_box_right_down_corner',
    'spoon_box_center', 'spoon_box_left_up_corner', 'spoon_box_right_up_corner', 'spoon_box_left_down_corner', 'spoon_box_right_down_corner',
    'bowl_box_center', 'bowl_box_left_up_corner', 'bowl_box_right_up_corner', 'bowl_box_left_down_corner', 'bowl_box_right_down_corner',
    'banana_box_center', 'banana_box_left_up_corner', 'banana_box_right_up_corner', 'banana_box_left_down_corner', 'banana_box_right_down_corner',
    'apple_box_center', 'apple_box_left_up_corner', 'apple_box_right_up_corner', 'apple_box_left_down_corner', 'apple_box_right_down_corner',
    'sandwich_box_center', 'sandwich_box_left_up_corner', 'sandwich_box_right_up_corner', 'sandwich_box_left_down_corner', 'sandwich_box_right_down_corner',
    'orange_box_center', 'orange_box_left_up_corner', 'orange_box_right_up_corner', 'orange_box_left_down_corner', 'orange_box_right_down_corner',
    'broccoli_box_center', 'broccoli_box_left_up_corner', 'broccoli_box_right_up_corner', 'broccoli_box_left_down_corner', 'broccoli_box_right_down_corner',
    'carrot_box_center', 'carrot_box_left_up_corner', 'carrot_box_right_up_corner', 'carrot_box_left_down_corner', 'carrot_box_right_down_corner',
    'hot dog_box_center', 'hot dog_box_left_up_corner', 'hot dog_box_right_up_corner', 'hot dog_box_left_down_corner', 'hot dog_box_right_down_corner',
    'pizza_box_center', 'pizza_box_left_up_corner', 'pizza_box_right_up_corner', 'pizza_box_left_down_corner', 'pizza_box_right_down_corner',
    'donut_box_center', 'donut_box_left_up_corner', 'donut_box_right_up_corner', 'donut_box_left_down_corner', 'donut_box_right_down_corner',
    'cake_box_center', 'cake_box_left_up_corner', 'cake_box_right_up_corner', 'cake_box_left_down_corner', 'cake_box_right_down_corner',
    'chair_box_center', 'chair_box_left_up_corner', 'chair_box_right_up_corner', 'chair_box_left_down_corner', 'chair_box_right_down_corner',
    'couch_box_center', 'couch_box_left_up_corner', 'couch_box_right_up_corner', 'couch_box_left_down_corner', 'couch_box_right_down_corner',
    'potted plant_box_center', 'potted plant_box_left_up_corner', 'potted plant_box_right_up_corner', 'potted plant_box_left_down_corner', 'potted plant_box_right_down_corner',
    'bed_box_center', 'bed_box_left_up_corner', 'bed_box_right_up_corner', 'bed_box_left_down_corner', 'bed_box_right_down_corner',
    'dining table_box_center', 'dining table_box_left_up_corner', 'dining table_box_right_up_corner', 'dining table_box_left_down_corner', 'dining table_box_right_down_corner',
    'toilet_box_center', 'toilet_box_left_up_corner', 'toilet_box_right_up_corner', 'toilet_box_left_down_corner', 'toilet_box_right_down_corner',
    'tv_box_center', 'tv_box_left_up_corner', 'tv_box_right_up_corner', 'tv_box_left_down_corner', 'tv_box_right_down_corner',
    'laptop_box_center', 'laptop_box_left_up_corner', 'laptop_box_right_up_corner', 'laptop_box_left_down_corner', 'laptop_box_right_down_corner',
    'mouse_box_center', 'mouse_box_left_up_corner', 'mouse_box_right_up_corner', 'mouse_box_left_down_corner', 'mouse_box_right_down_corner',
    'remote_box_center', 'remote_box_left_up_corner', 'remote_box_right_up_corner', 'remote_box_left_down_corner', 'remote_box_right_down_corner',
    'keyboard_box_center', 'keyboard_box_left_up_corner', 'keyboard_box_right_up_corner', 'keyboard_box_left_down_corner', 'keyboard_box_right_down_corner',
    'cell phone_box_center', 'cell phone_box_left_up_corner', 'cell phone_box_right_up_corner', 'cell phone_box_left_down_corner', 'cell phone_box_right_down_corner',
    'microwave_box_center', 'microwave_box_left_up_corner', 'microwave_box_right_up_corner', 'microwave_box_left_down_corner', 'microwave_box_right_down_corner',
    'oven_box_center', 'oven_box_left_up_corner', 'oven_box_right_up_corner', 'oven_box_left_down_corner', 'oven_box_right_down_corner',
    'toaster_box_center', 'toaster_box_left_up_corner', 'toaster_box_right_up_corner', 'toaster_box_left_down_corner', 'toaster_box_right_down_corner',
    'sink_box_center', 'sink_box_left_up_corner', 'sink_box_right_up_corner', 'sink_box_left_down_corner', 'sink_box_right_down_corner',
    'refrigerator_box_center', 'refrigerator_box_left_up_corner', 'refrigerator_box_right_up_corner', 'refrigerator_box_left_down_corner', 'refrigerator_box_right_down_corner',
    'book_box_center', 'book_box_left_up_corner', 'book_box_right_up_corner', 'book_box_left_down_corner', 'book_box_right_down_corner',
    'clock_box_center', 'clock_box_left_up_corner', 'clock_box_right_up_corner', 'clock_box_left_down_corner', 'clock_box_right_down_corner',
    'vase_box_center', 'vase_box_left_up_corner', 'vase_box_right_up_corner', 'vase_box_left_down_corner', 'vase_box_right_down_corner',
    'scissors_box_center', 'scissors_box_left_up_corner', 'scissors_box_right_up_corner', 'scissors_box_left_down_corner', 'scissors_box_right_down_corner',
    'teddy bear_box_center', 'teddy bear_box_left_up_corner', 'teddy bear_box_right_up_corner', 'teddy bear_box_left_down_corner', 'teddy bear_box_right_down_corner',
    'hair drier_box_center', 'hair drier_box_left_up_corner', 'hair drier_box_right_up_corner', 'hair drier_box_left_down_corner', 'hair drier_box_right_down_corner',
    'toothbrush_box_center', 'toothbrush_box_left_up_corner', 'toothbrush_box_right_up_corner', 'toothbrush_box_left_down_corner', 'toothbrush_box_right_down_corner',
]


COCO_UPRIGHT_POSE = np.array([
    [3.0, 5, 2.0], [0, 0, 2.0], [6, 0, 2.0], [0, 10, 2.0], [6, 10, 2.0],
])


HFLIP = {
    'person_box_left_up_corner': 'person_box_right_up_corner', 'person_box_right_up_corner': 'person_box_left_up_corner', 'person_box_left_down_corner': 'person_box_right_down_corner', 'person_box_right_down_corner': 'person_box_left_down_corner',
    # 'bicycle_box_left_up_corner': 'bicycle_box_right_up_corner', 'bicycle_box_right_up_corner': 'bicycle_box_left_up_corner', 'bicycle_box_left_down_corner': 'bicycle_box_right_down_corner', 'bicycle_box_right_down_corner': 'bicycle_box_left_down_corner',
    # 'car_box_left_up_corner': 'car_box_right_up_corner', 'car_box_right_up_corner': 'car_box_left_up_corner', 'car_box_left_down_corner': 'car_box_right_down_corner', 'car_box_right_down_corner': 'car_box_left_down_corner',
    # 'motorcycle_box_left_up_corner': 'motorcycle_box_right_up_corner', 'motorcycle_box_right_up_corner': 'motorcycle_box_left_up_corner', 'motorcycle_box_left_down_corner': 'motorcycle_box_right_down_corner', 'motorcycle_box_right_down_corner': 'motorcycle_box_left_down_corner',
    # 'airplane_box_left_up_corner': 'airplane_box_right_up_corner', 'airplane_box_right_up_corner': 'airplane_box_left_up_corner', 'airplane_box_left_down_corner': 'airplane_box_right_down_corner', 'airplane_box_right_down_corner': 'airplane_box_left_down_corner',
    # 'bus_box_left_up_corner': 'bus_box_right_up_corner', 'bus_box_right_up_corner': 'bus_box_left_up_corner', 'bus_box_left_down_corner': 'bus_box_right_down_corner', 'bus_box_right_down_corner': 'bus_box_left_down_corner',
    # 'train_box_left_up_corner': 'train_box_right_up_corner', 'train_box_right_up_corner': 'train_box_left_up_corner', 'train_box_left_down_corner': 'train_box_right_down_corner', 'train_box_right_down_corner': 'train_box_left_down_corner',
    # 'truck_box_left_up_corner': 'truck_box_right_up_corner', 'truck_box_right_up_corner': 'truck_box_left_up_corner', 'truck_box_left_down_corner': 'truck_box_right_down_corner', 'truck_box_right_down_corner': 'truck_box_left_down_corner',
    # 'boat_box_left_up_corner': 'boat_box_right_up_corner', 'boat_box_right_up_corner': 'boat_box_left_up_corner', 'boat_box_left_down_corner': 'boat_box_right_down_corner', 'boat_box_right_down_corner': 'boat_box_left_down_corner',
    # 'traffic light_box_left_up_corner': 'traffic light_box_right_up_corner', 'traffic light_box_right_up_corner': 'traffic light_box_left_up_corner', 'traffic light_box_left_down_corner': 'traffic light_box_right_down_corner', 'traffic light_box_right_down_corner': 'traffic light_box_left_down_corner',
    # 'fire hydrant_box_left_up_corner': 'fire hydrant_box_right_up_corner', 'fire hydrant_box_right_up_corner': 'fire hydrant_box_left_up_corner', 'fire hydrant_box_left_down_corner': 'fire hydrant_box_right_down_corner', 'fire hydrant_box_right_down_corner': 'fire hydrant_box_left_down_corner',
    # 'stop sign_box_left_up_corner': 'stop sign_box_right_up_corner', 'stop sign_box_right_up_corner': 'stop sign_box_left_up_corner', 'stop sign_box_left_down_corner': 'stop sign_box_right_down_corner', 'stop sign_box_right_down_corner': 'stop sign_box_left_down_corner',
    # 'parking meter_box_left_up_corner': 'parking meter_box_right_up_corner', 'parking meter_box_right_up_corner': 'parking meter_box_left_up_corner', 'parking meter_box_left_down_corner': 'parking meter_box_right_down_corner', 'parking meter_box_right_down_corner': 'parking meter_box_left_down_corner',
    # 'bench_box_left_up_corner': 'bench_box_right_up_corner', 'bench_box_right_up_corner': 'bench_box_left_up_corner', 'bench_box_left_down_corner': 'bench_box_right_down_corner', 'bench_box_right_down_corner': 'bench_box_left_down_corner',
    # 'bird_box_left_up_corner': 'bird_box_right_up_corner', 'bird_box_right_up_corner': 'bird_box_left_up_corner', 'bird_box_left_down_corner': 'bird_box_right_down_corner', 'bird_box_right_down_corner': 'bird_box_left_down_corner',
    # 'cat_box_left_up_corner': 'cat_box_right_up_corner', 'cat_box_right_up_corner': 'cat_box_left_up_corner', 'cat_box_left_down_corner': 'cat_box_right_down_corner', 'cat_box_right_down_corner': 'cat_box_left_down_corner',
    # 'dog_box_left_up_corner': 'dog_box_right_up_corner', 'dog_box_right_up_corner': 'dog_box_left_up_corner', 'dog_box_left_down_corner': 'dog_box_right_down_corner', 'dog_box_right_down_corner': 'dog_box_left_down_corner',
    # 'horse_box_left_up_corner': 'horse_box_right_up_corner', 'horse_box_right_up_corner': 'horse_box_left_up_corner', 'horse_box_left_down_corner': 'horse_box_right_down_corner', 'horse_box_right_down_corner': 'horse_box_left_down_corner',
    # 'sheep_box_left_up_corner': 'sheep_box_right_up_corner', 'sheep_box_right_up_corner': 'sheep_box_left_up_corner', 'sheep_box_left_down_corner': 'sheep_box_right_down_corner', 'sheep_box_right_down_corner': 'sheep_box_left_down_corner',
    # 'cow_box_left_up_corner': 'cow_box_right_up_corner', 'cow_box_right_up_corner': 'cow_box_left_up_corner', 'cow_box_left_down_corner': 'cow_box_right_down_corner', 'cow_box_right_down_corner': 'cow_box_left_down_corner',
    # 'elephant_box_left_up_corner': 'elephant_box_right_up_corner', 'elephant_box_right_up_corner': 'elephant_box_left_up_corner', 'elephant_box_left_down_corner': 'elephant_box_right_down_corner', 'elephant_box_right_down_corner': 'elephant_box_left_down_corner',
    # 'bear_box_left_up_corner': 'bear_box_right_up_corner', 'bear_box_right_up_corner': 'bear_box_left_up_corner', 'bear_box_left_down_corner': 'bear_box_right_down_corner', 'bear_box_right_down_corner': 'bear_box_left_down_corner',
    # 'zebra_box_left_up_corner': 'zebra_box_right_up_corner', 'zebra_box_right_up_corner': 'zebra_box_left_up_corner', 'zebra_box_left_down_corner': 'zebra_box_right_down_corner', 'zebra_box_right_down_corner': 'zebra_box_left_down_corner',
    # 'giraffe_box_left_up_corner': 'giraffe_box_right_up_corner', 'giraffe_box_right_up_corner': 'giraffe_box_left_up_corner', 'giraffe_box_left_down_corner': 'giraffe_box_right_down_corner', 'giraffe_box_right_down_corner': 'giraffe_box_left_down_corner',
    # 'backpack_box_left_up_corner': 'backpack_box_right_up_corner', 'backpack_box_right_up_corner': 'backpack_box_left_up_corner', 'backpack_box_left_down_corner': 'backpack_box_right_down_corner', 'backpack_box_right_down_corner': 'backpack_box_left_down_corner',
    # 'umbrella_box_left_up_corner': 'umbrella_box_right_up_corner', 'umbrella_box_right_up_corner': 'umbrella_box_left_up_corner', 'umbrella_box_left_down_corner': 'umbrella_box_right_down_corner', 'umbrella_box_right_down_corner': 'umbrella_box_left_down_corner',
    # 'handbag_box_left_up_corner': 'handbag_box_right_up_corner', 'handbag_box_right_up_corner': 'handbag_box_left_up_corner', 'handbag_box_left_down_corner': 'handbag_box_right_down_corner', 'handbag_box_right_down_corner': 'handbag_box_left_down_corner',
    # 'tie_box_left_up_corner': 'tie_box_right_up_corner', 'tie_box_right_up_corner': 'tie_box_left_up_corner', 'tie_box_left_down_corner': 'tie_box_right_down_corner', 'tie_box_right_down_corner': 'tie_box_left_down_corner',
    # 'suitcase_box_left_up_corner': 'suitcase_box_right_up_corner', 'suitcase_box_right_up_corner': 'suitcase_box_left_up_corner', 'suitcase_box_left_down_corner': 'suitcase_box_right_down_corner', 'suitcase_box_right_down_corner': 'suitcase_box_left_down_corner',
    # 'frisbee_box_left_up_corner': 'frisbee_box_right_up_corner', 'frisbee_box_right_up_corner': 'frisbee_box_left_up_corner', 'frisbee_box_left_down_corner': 'frisbee_box_right_down_corner', 'frisbee_box_right_down_corner': 'frisbee_box_left_down_corner',
    # 'skis_box_left_up_corner': 'skis_box_right_up_corner', 'skis_box_right_up_corner': 'skis_box_left_up_corner', 'skis_box_left_down_corner': 'skis_box_right_down_corner', 'skis_box_right_down_corner': 'skis_box_left_down_corner',
    # 'snowboard_box_left_up_corner': 'snowboard_box_right_up_corner', 'snowboard_box_right_up_corner': 'snowboard_box_left_up_corner', 'snowboard_box_left_down_corner': 'snowboard_box_right_down_corner', 'snowboard_box_right_down_corner': 'snowboard_box_left_down_corner',
    # 'sports ball_box_left_up_corner': 'sports ball_box_right_up_corner', 'sports ball_box_right_up_corner': 'sports ball_box_left_up_corner', 'sports ball_box_left_down_corner': 'sports ball_box_right_down_corner', 'sports ball_box_right_down_corner': 'sports ball_box_left_down_corner',
    # 'kite_box_left_up_corner': 'kite_box_right_up_corner', 'kite_box_right_up_corner': 'kite_box_left_up_corner', 'kite_box_left_down_corner': 'kite_box_right_down_corner', 'kite_box_right_down_corner': 'kite_box_left_down_corner',
    # 'baseball bat_box_left_up_corner': 'baseball bat_box_right_up_corner', 'baseball bat_box_right_up_corner': 'baseball bat_box_left_up_corner', 'baseball bat_box_left_down_corner': 'baseball bat_box_right_down_corner', 'baseball bat_box_right_down_corner': 'baseball bat_box_left_down_corner',
    # 'baseball glove_box_left_up_corner': 'baseball glove_box_right_up_corner', 'baseball glove_box_right_up_corner': 'baseball glove_box_left_up_corner', 'baseball glove_box_left_down_corner': 'baseball glove_box_right_down_corner', 'baseball glove_box_right_down_corner': 'baseball glove_box_left_down_corner',
    # 'skateboard_box_left_up_corner': 'skateboard_box_right_up_corner', 'skateboard_box_right_up_corner': 'skateboard_box_left_up_corner', 'skateboard_box_left_down_corner': 'skateboard_box_right_down_corner', 'skateboard_box_right_down_corner': 'skateboard_box_left_down_corner',
    # 'surfboard_box_left_up_corner': 'surfboard_box_right_up_corner', 'surfboard_box_right_up_corner': 'surfboard_box_left_up_corner', 'surfboard_box_left_down_corner': 'surfboard_box_right_down_corner', 'surfboard_box_right_down_corner': 'surfboard_box_left_down_corner',
    # 'tennis racket_box_left_up_corner': 'tennis racket_box_right_up_corner', 'tennis racket_box_right_up_corner': 'tennis racket_box_left_up_corner', 'tennis racket_box_left_down_corner': 'tennis racket_box_right_down_corner', 'tennis racket_box_right_down_corner': 'tennis racket_box_left_down_corner',
    # 'bottle_box_left_up_corner': 'bottle_box_right_up_corner', 'bottle_box_right_up_corner': 'bottle_box_left_up_corner', 'bottle_box_left_down_corner': 'bottle_box_right_down_corner', 'bottle_box_right_down_corner': 'bottle_box_left_down_corner',
    # 'wine glass_box_left_up_corner': 'wine glass_box_right_up_corner', 'wine glass_box_right_up_corner': 'wine glass_box_left_up_corner', 'wine glass_box_left_down_corner': 'wine glass_box_right_down_corner', 'wine glass_box_right_down_corner': 'wine glass_box_left_down_corner',
    # 'cup_box_left_up_corner': 'cup_box_right_up_corner', 'cup_box_right_up_corner': 'cup_box_left_up_corner', 'cup_box_left_down_corner': 'cup_box_right_down_corner', 'cup_box_right_down_corner': 'cup_box_left_down_corner',
    # 'fork_box_left_up_corner': 'fork_box_right_up_corner', 'fork_box_right_up_corner': 'fork_box_left_up_corner', 'fork_box_left_down_corner': 'fork_box_right_down_corner', 'fork_box_right_down_corner': 'fork_box_left_down_corner',
    # 'knife_box_left_up_corner': 'knife_box_right_up_corner', 'knife_box_right_up_corner': 'knife_box_left_up_corner', 'knife_box_left_down_corner': 'knife_box_right_down_corner', 'knife_box_right_down_corner': 'knife_box_left_down_corner',
    # 'spoon_box_left_up_corner': 'spoon_box_right_up_corner', 'spoon_box_right_up_corner': 'spoon_box_left_up_corner', 'spoon_box_left_down_corner': 'spoon_box_right_down_corner', 'spoon_box_right_down_corner': 'spoon_box_left_down_corner',
    # 'bowl_box_left_up_corner': 'bowl_box_right_up_corner', 'bowl_box_right_up_corner': 'bowl_box_left_up_corner', 'bowl_box_left_down_corner': 'bowl_box_right_down_corner', 'bowl_box_right_down_corner': 'bowl_box_left_down_corner',
    # 'banana_box_left_up_corner': 'banana_box_right_up_corner', 'banana_box_right_up_corner': 'banana_box_left_up_corner', 'banana_box_left_down_corner': 'banana_box_right_down_corner', 'banana_box_right_down_corner': 'banana_box_left_down_corner',
    # 'apple_box_left_up_corner': 'apple_box_right_up_corner', 'apple_box_right_up_corner': 'apple_box_left_up_corner', 'apple_box_left_down_corner': 'apple_box_right_down_corner', 'apple_box_right_down_corner': 'apple_box_left_down_corner',
    # 'sandwich_box_left_up_corner': 'sandwich_box_right_up_corner', 'sandwich_box_right_up_corner': 'sandwich_box_left_up_corner', 'sandwich_box_left_down_corner': 'sandwich_box_right_down_corner', 'sandwich_box_right_down_corner': 'sandwich_box_left_down_corner',
    # 'orange_box_left_up_corner': 'orange_box_right_up_corner', 'orange_box_right_up_corner': 'orange_box_left_up_corner', 'orange_box_left_down_corner': 'orange_box_right_down_corner', 'orange_box_right_down_corner': 'orange_box_left_down_corner',
    # 'broccoli_box_left_up_corner': 'broccoli_box_right_up_corner', 'broccoli_box_right_up_corner': 'broccoli_box_left_up_corner', 'broccoli_box_left_down_corner': 'broccoli_box_right_down_corner', 'broccoli_box_right_down_corner': 'broccoli_box_left_down_corner',
    # 'carrot_box_left_up_corner': 'carrot_box_right_up_corner', 'carrot_box_right_up_corner': 'carrot_box_left_up_corner', 'carrot_box_left_down_corner': 'carrot_box_right_down_corner', 'carrot_box_right_down_corner': 'carrot_box_left_down_corner',
    # 'hot dog_box_left_up_corner': 'hot dog_box_right_up_corner', 'hot dog_box_right_up_corner': 'hot dog_box_left_up_corner', 'hot dog_box_left_down_corner': 'hot dog_box_right_down_corner', 'hot dog_box_right_down_corner': 'hot dog_box_left_down_corner',
    # 'pizza_box_left_up_corner': 'pizza_box_right_up_corner', 'pizza_box_right_up_corner': 'pizza_box_left_up_corner', 'pizza_box_left_down_corner': 'pizza_box_right_down_corner', 'pizza_box_right_down_corner': 'pizza_box_left_down_corner',
    # 'donut_box_left_up_corner': 'donut_box_right_up_corner', 'donut_box_right_up_corner': 'donut_box_left_up_corner', 'donut_box_left_down_corner': 'donut_box_right_down_corner', 'donut_box_right_down_corner': 'donut_box_left_down_corner',
    # 'cake_box_left_up_corner': 'cake_box_right_up_corner', 'cake_box_right_up_corner': 'cake_box_left_up_corner', 'cake_box_left_down_corner': 'cake_box_right_down_corner', 'cake_box_right_down_corner': 'cake_box_left_down_corner',
    # 'chair_box_left_up_corner': 'chair_box_right_up_corner', 'chair_box_right_up_corner': 'chair_box_left_up_corner', 'chair_box_left_down_corner': 'chair_box_right_down_corner', 'chair_box_right_down_corner': 'chair_box_left_down_corner',
    # 'couch_box_left_up_corner': 'couch_box_right_up_corner', 'couch_box_right_up_corner': 'couch_box_left_up_corner', 'couch_box_left_down_corner': 'couch_box_right_down_corner', 'couch_box_right_down_corner': 'couch_box_left_down_corner',
    # 'potted plant_box_left_up_corner': 'potted plant_box_right_up_corner', 'potted plant_box_right_up_corner': 'potted plant_box_left_up_corner', 'potted plant_box_left_down_corner': 'potted plant_box_right_down_corner', 'potted plant_box_right_down_corner': 'potted plant_box_left_down_corner',
    # 'bed_box_left_up_corner': 'bed_box_right_up_corner', 'bed_box_right_up_corner': 'bed_box_left_up_corner', 'bed_box_left_down_corner': 'bed_box_right_down_corner', 'bed_box_right_down_corner': 'bed_box_left_down_corner',
    # 'dining table_box_left_up_corner': 'dining table_box_right_up_corner', 'dining table_box_right_up_corner': 'dining table_box_left_up_corner', 'dining table_box_left_down_corner': 'dining table_box_right_down_corner', 'dining table_box_right_down_corner': 'dining table_box_left_down_corner',
    # 'toilet_box_left_up_corner': 'toilet_box_right_up_corner', 'toilet_box_right_up_corner': 'toilet_box_left_up_corner', 'toilet_box_left_down_corner': 'toilet_box_right_down_corner', 'toilet_box_right_down_corner': 'toilet_box_left_down_corner',
    # 'tv_box_left_up_corner': 'tv_box_right_up_corner', 'tv_box_right_up_corner': 'tv_box_left_up_corner', 'tv_box_left_down_corner': 'tv_box_right_down_corner', 'tv_box_right_down_corner': 'tv_box_left_down_corner',
    # 'laptop_box_left_up_corner': 'laptop_box_right_up_corner', 'laptop_box_right_up_corner': 'laptop_box_left_up_corner', 'laptop_box_left_down_corner': 'laptop_box_right_down_corner', 'laptop_box_right_down_corner': 'laptop_box_left_down_corner',
    # 'mouse_box_left_up_corner': 'mouse_box_right_up_corner', 'mouse_box_right_up_corner': 'mouse_box_left_up_corner', 'mouse_box_left_down_corner': 'mouse_box_right_down_corner', 'mouse_box_right_down_corner': 'mouse_box_left_down_corner',
    # 'remote_box_left_up_corner': 'remote_box_right_up_corner', 'remote_box_right_up_corner': 'remote_box_left_up_corner', 'remote_box_left_down_corner': 'remote_box_right_down_corner', 'remote_box_right_down_corner': 'remote_box_left_down_corner',
    # 'keyboard_box_left_up_corner': 'keyboard_box_right_up_corner', 'keyboard_box_right_up_corner': 'keyboard_box_left_up_corner', 'keyboard_box_left_down_corner': 'keyboard_box_right_down_corner', 'keyboard_box_right_down_corner': 'keyboard_box_left_down_corner',
    # 'cell phone_box_left_up_corner': 'cell phone_box_right_up_corner', 'cell phone_box_right_up_corner': 'cell phone_box_left_up_corner', 'cell phone_box_left_down_corner': 'cell phone_box_right_down_corner', 'cell phone_box_right_down_corner': 'cell phone_box_left_down_corner',
    # 'microwave_box_left_up_corner': 'microwave_box_right_up_corner', 'microwave_box_right_up_corner': 'microwave_box_left_up_corner', 'microwave_box_left_down_corner': 'microwave_box_right_down_corner', 'microwave_box_right_down_corner': 'microwave_box_left_down_corner',
    # 'oven_box_left_up_corner': 'oven_box_right_up_corner', 'oven_box_right_up_corner': 'oven_box_left_up_corner', 'oven_box_left_down_corner': 'oven_box_right_down_corner', 'oven_box_right_down_corner': 'oven_box_left_down_corner',
    # 'toaster_box_left_up_corner': 'toaster_box_right_up_corner', 'toaster_box_right_up_corner': 'toaster_box_left_up_corner', 'toaster_box_left_down_corner': 'toaster_box_right_down_corner', 'toaster_box_right_down_corner': 'toaster_box_left_down_corner',
    # 'sink_box_left_up_corner': 'sink_box_right_up_corner', 'sink_box_right_up_corner': 'sink_box_left_up_corner', 'sink_box_left_down_corner': 'sink_box_right_down_corner', 'sink_box_right_down_corner': 'sink_box_left_down_corner',
    # 'refrigerator_box_left_up_corner': 'refrigerator_box_right_up_corner', 'refrigerator_box_right_up_corner': 'refrigerator_box_left_up_corner', 'refrigerator_box_left_down_corner': 'refrigerator_box_right_down_corner', 'refrigerator_box_right_down_corner': 'refrigerator_box_left_down_corner',
    # 'book_box_left_up_corner': 'book_box_right_up_corner', 'book_box_right_up_corner': 'book_box_left_up_corner', 'book_box_left_down_corner': 'book_box_right_down_corner', 'book_box_right_down_corner': 'book_box_left_down_corner',
    # 'clock_box_left_up_corner': 'clock_box_right_up_corner', 'clock_box_right_up_corner': 'clock_box_left_up_corner', 'clock_box_left_down_corner': 'clock_box_right_down_corner', 'clock_box_right_down_corner': 'clock_box_left_down_corner',
    # 'vase_box_left_up_corner': 'vase_box_right_up_corner', 'vase_box_right_up_corner': 'vase_box_left_up_corner', 'vase_box_left_down_corner': 'vase_box_right_down_corner', 'vase_box_right_down_corner': 'vase_box_left_down_corner',
    # 'scissors_box_left_up_corner': 'scissors_box_right_up_corner', 'scissors_box_right_up_corner': 'scissors_box_left_up_corner', 'scissors_box_left_down_corner': 'scissors_box_right_down_corner', 'scissors_box_right_down_corner': 'scissors_box_left_down_corner',
    # 'teddy bear_box_left_up_corner': 'teddy bear_box_right_up_corner', 'teddy bear_box_right_up_corner': 'teddy bear_box_left_up_corner', 'teddy bear_box_left_down_corner': 'teddy bear_box_right_down_corner', 'teddy bear_box_right_down_corner': 'teddy bear_box_left_down_corner',
    # 'hair drier_box_left_up_corner': 'hair drier_box_right_up_corner', 'hair drier_box_right_up_corner': 'hair drier_box_left_up_corner', 'hair drier_box_left_down_corner': 'hair drier_box_right_down_corner', 'hair drier_box_right_down_corner': 'hair drier_box_left_down_corner',
    # 'toothbrush_box_left_up_corner': 'toothbrush_box_right_up_corner', 'toothbrush_box_right_up_corner': 'toothbrush_box_left_up_corner', 'toothbrush_box_left_down_corner': 'toothbrush_box_right_down_corner', 'toothbrush_box_right_down_corner': 'toothbrush_box_left_down_corner',
}

COCO_BOX_SIGMAS = [
    0.03, 0.03, 0.03, 0.03, 0.03, # center, left_up_corner, right_up_corner, left_down_corner, right_down_corner
]


COCO_BOX_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(COCO_KEYPOINTS) - 3)


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                skeleton=COCO_BOX_SKELETON,
                                score_weights=COCO_BOX_SCORE_WEIGHTS)
    ann.set(pose, np.array(COCO_BOX_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_coco.png') as ax:
        keypoint_painter.annotation(ax, ann)

    ann_kin = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                    skeleton=KINEMATIC_TREE_SKELETON,
                                    score_weights=COCO_BOX_SCORE_WEIGHTS)
    ann_kin.set(pose, np.array(COCO_BOX_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann_kin, filename='docs/skeleton_kinematic_tree.png') as ax:
        keypoint_painter.annotation(ax, ann_kin)
    '''
    ann_dense = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                      skeleton=DENSER_COCO_BOX_SKELETON,
                                      score_weights=COCO_PERSON_SCORE_WEIGHTS)
    ann_dense.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, ann_bg=ann_dense, filename='docs/skeleton_dense.png') as ax:
        keypoint_painter.annotation(ax, ann_dense)
    '''


def print_associations():
    for j1, j2 in COCO_BOX_SKELETON:
        print(COCO_KEYPOINTS[j1 - 1], '-', COCO_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(COCO_UPRIGHT_POSE)
