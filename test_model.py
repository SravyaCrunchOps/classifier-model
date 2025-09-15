import joblib
import pandas as pd


# Get the features for prediction
def create_animal():
    print("Enter features here: ")
    get_hair = input("hair(0 or 1): ")
    get_feathers = input("feathers(0 or 1): ")
    get_eggs = input("eggs(0 or 1): ")
    get_milk = input("milk(0 or 1): ")
    get_airborne = input("airborne(0 or 1): ")
    get_aquatic = input("aquatic(0 or 1): ")
    get_predator = input("predator(0 or 1: ")
    get_toothed = input("toothed(0 or 1): ")
    get_backbone = input("backbone(0 or 1): ")
    get_breathes = input("breathes(0 or 1): ")
    get_venomous = input("venomous(0 or 1): ")
    get_fins = input("fins(0 or 1): ")
    get_legs = input("legs(0,2,4,5,6,8): ")
    get_tail = input("tail(0 or 1): ")
    get_domestic = input("domestic(0 or 1): ")
    get_catsize = input("catsize(0 or 1): ")
    features_dict = {
        'hair': int(get_hair), 'feathers': int(get_feathers), 'eggs': int(get_eggs), 'milk': int(get_milk), 
        'airborne': int(get_airborne), 'aquatic': int(get_aquatic), 'predator': int(get_predator), 'toothed': int(get_toothed),
        'backbone': int(get_backbone), 'breathes': int(get_breathes), 'venomous': int(get_venomous), 'fins': int(get_fins),
        'legs': int(get_legs), 'tail': int(get_tail), 'domestic': int(get_domestic), 'catsize': int(get_catsize)
    }
    # print(features_dict)
    return features_dict




def predict_animal(animal_name):
    #  load model
    try:
        model = joblib.load("./zoo_animal_classifier_model.pkl")
        # print(model)
    except Exception as e:
        print(e)

    # load dataset to get animal_name features
    df = pd.read_csv("./zoo_animals_data.csv")
    X = df.drop(columns=['animal_name', 'class_type', 'class_name'])
    df_columns = X.columns.to_list()

    is_feature = df[df['animal_name'] == animal_name]

    if is_feature.empty:
        print("""This animal is not found in dataset. To know this animal give me full details such as:
        features_dict: { 'hair', 'feathers', 'milk', ........, 'domestic', 'catsize'}       
        """)
        feature = create_animal()
        input = pd.DataFrame([feature])[df_columns]

    else: 
        input = is_feature.drop(columns=['animal_name', 'class_type', 'class_name'])

    # print(input)

    y_result = model.predict(input)[0]
    print(f"\nAnimal Type: {y_result}\n")


if __name__ == "__main__":
    predict_animal('boar')

