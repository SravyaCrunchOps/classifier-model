import joblib
import pandas as pd


MODEL_PATH = '../models/classifier_model.pkl'
FEATURE_PATH = '../feature_store/feature_names.pkl'
PREPROCESSED_DATASET = '../feature_store/preprocessed_data.csv'

# Get the features for prediction
def create_animal():
    print("Enter features here: ")
    get_hair = input("hair(0 or 1): ")
    get_eggs = input("eggs(0 or 1): ")
    get_milk = input("milk(0 or 1): ")
    get_predator = input("predator(0 or 1: ")
    get_toothed = input("toothed(0 or 1): ")
    get_backbone = input("backbone(0 or 1): ")
    get_breathes = input("breathes(0 or 1): ")
    get_venomous = input("venomous(0 or 1): ")
    get_legs = input("legs(0,2,4,5,6,8): ")
    get_tail = input("tail(0 or 1): ")
    can_fly = input("can_fly(0 or 1): ")
    can_swim = input("can_swim(0 or 1): ")
    is_domestic_pet = input("is_domestic_pet(0 or 1): ")
    features_dict = {
        'hair': int(get_hair), 'eggs': int(get_eggs), 'milk': int(get_milk), 
        'predator': int(get_predator), 'toothed': int(get_toothed),
        'backbone': int(get_backbone), 'breathes': int(get_breathes), 
        'can_fly': int(can_fly), 'can_swim': int(can_swim), 'is_domestic_pet': int(is_domestic_pet),
        'venomous': int(get_venomous), 'legs': int(get_legs), 'tail': int(get_tail), 
    }
    # print(features_dict)
    return features_dict



def predict_animal(animal_name):
    #  load model
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)
        df = pd.read_csv(PREPROCESSED_DATASET)
        print("✅ model loaded")
    except Exception as e:
        print(e)

    # load dataset to get animal_name features
    feature = df[df['animal_name'] == animal_name]
    print(feature_names)

    if feature.empty:
        print("This animal is not found in dataset.")
        feature = create_animal()
        input = pd.DataFrame([feature])[feature_names]

    else: 
        input = feature.drop(columns=['animal_name', 'class_type', 'class_name', 'airborne', 'feathers', 'domestic', 'catsize', 'aquatic', 'fins'])

    # print(input)
    y_result = model.predict(input)[0]
    print(f"\nAnimal Type for {animal_name}: {y_result}\n")



if __name__ == "__main__":
    while True:
        animal_name = input("Enter animal name: ")
        if animal_name.lower() in ['exit', 'quit']:
            print("👋🏻 Bye")
            break
        predict_animal(animal_name)

