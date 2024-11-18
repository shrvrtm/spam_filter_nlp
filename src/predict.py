def predict_spam_line(model, vectorizer, text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

def predict_spam(model, vectorizer,input_path,output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as input_path:
            lines = input_path.readlines()

        spam = []
        ham = []

        with open(output_path, 'w', encoding='utf-8') as output_path:
            for line in lines:
                predict = predict_spam_line(model, vectorizer,line)
                if predict == 1:
                    spam.append(line)
                else:
                    ham.append(line)
            output_path.write("Ham:\n")
            for line in ham:
                output_path.write(line)

            output_path.write("\nSpam:\n")
            for line in spam:
                output_path.write(line)


    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_path}' не найден.")
    except IOError as e:
        print(f"Ошибка ввода-вывода: {e}")