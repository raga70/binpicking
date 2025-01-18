from main import classify_object

detected_items = []
for i in range(159):
    image = "affixobjects/output_objects/object_" + str(i+1) + "_RGB.png"
    result = classify_object(image)
    if result != "No object detected":
        detected_items.append(str(i) + ": " + result)

for item in detected_items:
    print(item)