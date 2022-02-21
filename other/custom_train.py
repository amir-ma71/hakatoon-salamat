import pandas as pd
import json
import numpy as np
export = pd.read_csv("./data/output/exportBERT_model-max250-0.85(90).csv")

dataset = pd.read_parquet("./data/test_for_submission.parquet")

vector_list = []
label_list = []
category_list = ['cleaning', 'book-student-literature', 'reptile', 'traditional', 'drums-percussion',
                 'leisure-hobbies-toys', 'winter-sports', 'stationery', 'transport', 'desktops', 'childrens-furniture',
                 'parts-accessories', 'mobile-phone-repair', 'utility', 'cat', 'electronic-devices',
                 'hobby-collectibles', 'dishwasher', 'car-and-motor', 'computer-and-it', 'building-and-garden',
                 'game-consoles-and-video-games', 'services-sports', 'cookware-tableware', 'magazines', 'offices',
                 'accessories', 'events-sports', 'laptops', 'washer-dryer', 'research', 'teaching', 'ball-sports',
                 'child-car-seat', 'dog', 'beauty-and-haircare', 'garden-and-landscaping', 'office-rent', 'presell',
                 'storage', 'carpets', 'plot-old', 'motorcycles', 'light', 'movies-and-music', 'jobs',
                 'tables-and-chairs', 'historical-objects', 'audio-video', 'office-sell', 'camping-outdoor',
                 'shoes-belt-bag', 'consulting', 'foreign-language', 'diving-watersports', 'shop-and-cash', 'fish',
                 'lost-things', 'mobile-phones', 'printer-scaner-copier', 'personal', 'house-villa-sell',
                 'parts-and-accessories', 'utensils-and-appliances', 'suite-apartment', 'farm-animals',
                 'childrens-clothing-and-shoe', 'accounting-finance-legal', 'fishing', 'wind', 'cafe-and-restaurant',
                 'watches', 'microwave-stove', 'hosting-and-web-design', 'volunteers', 'clothing', 'horses-equestrian',
                 'rodents-rabbits', 'shop-sell', 'garden-tools', 'sales-marketing', 'violins', 'classic',
                 'media-advertising', 'domains-and-websites', 'lost-animals', 'concert', 'sim-card',
                 'accounting-and-finance', 'boat', 'tv-and-stereo-furniture', 'software', 'bus-metro-train',
                 'theatre-and-cinema', 'entertainment', 'tickets-sports', 'art', 'tablet', 'kitchen', 'bicycle',
                 'modem-and-network-equipment', 'care-health-beauty', 'musical-instruments', 'for-sale',
                 'broadband-internet-service', 'instrument-cleaning-tailoring', 'barbershop-and-beautysalon', 'school',
                 'rhinestones', 'phone', 'religious', 'educational', 'apartment-rent', 'janitorial-cleaning',
                 'vehicles', 'personal-toys', 'travel-packages', 'computer-hardware-and-software', 'stereo-surround',
                 'lighting', 'video-dvdplayer', 'construction-craft', 'rental', 'coin-stamp', 'house-villa-rent',
                 'villa', 'education', 'industrial', 'sofa-armchair', 'heavy', 'partnership', 'camera-camcoders',
                 'textile-ornaments', 'CCTV', 'batch', 'medical-equipment', 'piano-keyboard', 'businesses',
                 'antiques-and-art', 'computer-and-mobile', 'industry-agriculture-business-rent',
                 'industrial-technology', 'equipments-and-machinery', 'leisure-hobbies', 'guitar-bass-amplifier',
                 'gift-certificate', 'literary', 'shop-restaurant', 'strollers-and-accessories', 'garden-and-patio',
                 'sport', 'health-beauty', 'apartment-sell', 'mp3-player', 'jewelry', 'bathroom-wc-sauna',
                 'industry-agriculture-business-sell', 'catering', 'repair-tool', 'mobile-tablet-accessories',
                 'historical', 'conference-meeting', 'fridge-and-freezer', 'beds-bedroom', 'birds', 'stove-and-heating',
                 'craftsmen', 'services', 'tv-projector', 'administration-and-hr', 'jewelry-and-watches', 'event',
                 'transportation', 'training', 'shop-rent'
                 ]
c = 0
for row in dataset.iterrows():
    vector = []

    category = json.loads(row[1]["post_data"])["category"]
    # 1 feature == category id
    if category not in category_list:
        category_list.append(category)
    vector.append(category_list.index(category))
    # 2 feature == len title
    vector.append(len(json.loads(row[1]["post_data"])["title"].split()))
    # 3 feature == len description
    vector.append(len(json.loads(row[1]["post_data"])["description"].split()))
    # 4 feature == num images
    vector.append(json.loads(row[1]["post_data"])["num_images"])
    # 5 feature == hide phone
    if json.loads(row[1]["post_data"])["contact"]["hide_phone"]:
        vector.append(1)
    else:
        vector.append(0)
    # 6 feature == len messages
    vector.append(len(json.loads(row[1]["messages"])))
    # 7 feature == is reported
    vector.append(row[1]["is_reported"])

    # 8 feature == bert predicted
    pred = export[export["unique_conversation_id"]==row[1]["unique_conversation_id"]]["predictions"].reset_index(drop=True)[0]
    vector.append(pred)
    # add label
    # if row[1]["label"]:
    #     label_list.append(np.array([0,1]))
    # else:
    #     label_list.append(np.array([1,0]))

    vector_list.append(vector)

    print(c)
    c += 1

final = vector_list

import pickle

with open('feature_vector_test.pickle', 'wb') as handle:
    pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)