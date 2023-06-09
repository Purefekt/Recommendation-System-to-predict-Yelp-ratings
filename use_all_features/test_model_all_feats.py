"""
Method Description: I used a model-based recommendation system. I have used the XGBoost model to predict the ratings.
I extracted the following features for each business:
From the business.json file I extracted -> All the features. For the single value features which which have boolean values, i set 1 for True and 0 for False. For features which were categorical, i converted each category into a 32bit deterministic hash value. I used one hot encoding to convert each category into a feature.
From the checkIn.json file I extracted -> the number of different check-ins done for the business.
From the photo.json file I extracted -> the number of photos for each business.
From the tip.json file I extracted -> the number of tips for each business
I extracted the following features for each user:
From the tip.json file I extracted -> the number of tips left by each user.
From the user.json file I extracted -> review count, yelping since which is the Linux timestamp value of when they created their account, number of friends, number of useful, number of funny, number of cool, number of fans, number of times this user was elite, average stars, compliment hot, compliment more, compliment profile, compliment cute, compliment cute, compliment list, compliment note, compliment plain, compliment cool, compliment funny, compliment writer, compliment photos.
Then I combined the data to create a (user_id, business_id) [feature1, feature2, ...]
This way, i created feature vectors of size 1423 which was 1423 different features.
I trained the model locally and saved the model.sav file
I used different hyperparameters and using hyperparameter tuning, was able to get the lowest RMSE value.

Error Distribution:
>=0 and <1: 102253
>=1 and <2: 32870
>=2 and <3: 6137
>=3 and <4: 783
>=4 1

RMSE:


Execution Time:
114s
"""

from pyspark.context import SparkContext
import json
from datetime import datetime
import pytz
import time
import sys
import ast
from datetime import datetime
import numpy as np
import decimal
import pickle

start_time = time.time()

sc = SparkContext()
sc.setLogLevel('ERROR')

# FOLDER_PATH = sys.argv[1]
# TESTING_FILE_PATH = sys.argv[2]
# OUTPUT_FILE_PATH = sys.argv[3]

FOLDER_PATH = '/Users/veersingh/Desktop/competition_files/'
TESTING_FILE_PATH = '/Users/veersingh/Desktop/competition_files/yelp_val.csv'
OUTPUT_FILE_PATH = '/Users/veersingh/Desktop/Recommendation-System-to-predict-Yelp-ratings/output.csv'

BUSINESS_FILE_PATH = FOLDER_PATH + 'business.json'
CHECKIN_FILE_PATH = FOLDER_PATH + 'checkin.json'
PHOTO_FILE_PATH = FOLDER_PATH + 'photo.json'
TIP_FILE_PATH = FOLDER_PATH + 'tip.json'
USER_FILE_PATH = FOLDER_PATH + 'user.json'

MODEL_FILE_PATH = 'model.sav'
LOADED_MODEL = pickle.load(open(MODEL_FILE_PATH, 'rb'))

# All the unique features for a business mined from business.json
ALL_FEATURES = {'& Probates': 0, '3D Printing': 1, 'ATV Rentals/Tours': 2, 'Acai Bowls': 3, 'Accessories': 4, 'Accountants': 5, 'Acne Treatment': 6, 'Active Life': 7, 'Acupuncture': 8, 'Addiction Medicine': 9, 'Adoption Services': 10, 'Adult': 11, 'Adult Education': 12, 'Adult Entertainment': 13, 'Advertising': 14, 'Aerial Fitness': 15, 'Aerial Tours': 16, 'Aestheticians': 17, 'Afghan': 18, 'African': 19, 'Air Duct Cleaning': 20, 'Aircraft Dealers': 21, 'Aircraft Repairs': 22, 'Airlines': 23, 'Airport Lounges': 24, 'Airport Shuttles': 25, 'Airport Terminals': 26, 'Airports': 27, 'Airsoft': 28, 'Allergists': 29, 'Alternative Medicine': 30, 'Amateur Sports Teams': 31, 'American (New)': 32, 'American (Traditional)': 33, 'Amusement Parks': 34, 'Anesthesiologists': 35, 'Animal Assisted Therapy': 36, 'Animal Physical Therapy': 37, 'Animal Shelters': 38, 'Antiques': 39, 'Apartment Agents': 40, 'Apartments': 41, 'Appliances': 42, 'Appliances & Repair': 43, 'Appraisal Services': 44, 'Aquarium Services': 45, 'Aquariums': 46, 'Arabian': 47, 'Arcades': 48, 'Archery': 49, 'Architects': 50, 'Architectural Tours': 51, 'Argentine': 52, 'Armenian': 53, 'Art Classes': 54, 'Art Galleries': 55, 'Art Museums': 56, 'Art Restoration': 57, 'Art Schools': 58, 'Art Space Rentals': 59, 'Art Supplies': 60, 'Art Tours': 61, 'Artificial Turf': 62, 'Arts & Crafts': 63, 'Arts & Entertainment': 64, 'Asian Fusion': 65, 'Assisted Living Facilities': 66, 'Astrologers': 67, 'Attraction Farms': 68, 'Auction Houses': 69, 'Audio/Visual Equipment Rental': 70, 'Audiologist': 71, 'Australian': 72, 'Austrian': 73, 'Auto Customization': 74, 'Auto Detailing': 75, 'Auto Electric Services': 76, 'Auto Glass Services': 77, 'Auto Insurance': 78, 'Auto Loan Providers': 79, 'Auto Parts & Supplies': 80, 'Auto Repair': 81, 'Auto Security': 82, 'Auto Upholstery': 83, 'Automotive': 84, 'Aviation Services': 85, 'Awnings': 86, 'Axe Throwing': 87, 'Ayurveda': 88, 'Baby Gear & Furniture': 89, 'Backflow Services': 90, 'Backshop': 91, 'Badminton': 92, 'Bagels': 93, 'Baguettes': 94, 'Bail Bondsmen': 95, 'Bakeries': 96, 'Balloon Services': 97, 'Bangladeshi': 98, 'Bankruptcy Law': 99, 'Banks & Credit Unions': 100, 'Bar Crawl': 101, 'Barbeque': 102, 'Barbers': 103, 'Barre Classes': 104, 'Bars': 105, 'Bartenders': 106, 'Bartending Schools': 107, 'Baseball Fields': 108, 'Basketball Courts': 109, 'Basque': 110, 'Battery Stores': 111, 'Batting Cages': 112, 'Bavarian': 113, 'Beach Bars': 114, 'Beach Equipment Rentals': 115, 'Beach Volleyball': 116, 'Beaches': 117, 'Beauty & Spas': 118, 'Bed & Breakfast': 119, 'Beer': 120, 'Beer Bar': 121, 'Beer Garden': 122, 'Beer Gardens': 123, 'Beer Hall': 124, 'Beer Tours': 125, 'Behavior Analysts': 126, 'Belgian': 127, 'Bespoke Clothing': 128, 'Beverage Store': 129, 'Bicycles': 130, 'Bike Parking': 131, 'Bike Rentals': 132, 'Bike Repair': 133, 'Bike Repair/Maintenance': 134, 'Bike Sharing': 135, 'Bike Shop': 136, 'Bike tours': 137, 'Bikes': 138, 'Billing Services': 139, 'Bingo Halls': 140, 'Bird Shops': 141, 'Bistros': 142, 'Blood & Plasma Donation Centers': 143, 'Blow Dry/Out Services': 144, 'Boat Charters': 145, 'Boat Dealers': 146, 'Boat Parts & Supplies': 147, 'Boat Repair': 148, 'Boat Tours': 149, 'Boating': 150, 'Body Contouring': 151, 'Body Shops': 152, 'Bookbinding': 153, 'Bookkeepers': 154, 'Books': 155, 'Bookstores': 156, 'Boot Camps': 157, 'Botanical Gardens': 158, 'Boudoir Photography': 159, 'Bounce House Rentals': 160, 'Bowling': 161, 'Boxing': 162, 'Brasseries': 163, 'Brazilian': 164, 'Brazilian Jiu-jitsu': 165, 'Breakfast & Brunch': 166, 'Breweries': 167, 'Brewing Supplies': 168, 'Brewpubs': 169, 'Bridal': 170, 'British': 171, 'Bubble Soccer': 172, 'Bubble Tea': 173, 'Buddhist Temples': 174, 'Buffets': 175, 'Building Supplies': 176, 'Bulgarian': 177, 'Burgers': 178, 'Burmese': 179, 'Bus Rental': 180, 'Bus Tours': 181, 'Buses': 182, 'Business Consulting': 183, 'Business Financing': 184, 'Business Law': 185, 'Butcher': 186, 'CPR Classes': 187, 'CSA': 188, 'Cabaret': 189, 'Cabinetry': 190, 'Cafes': 191, 'Cafeteria': 192, 'Cajun/Creole': 193, 'Calligraphy': 194, 'Cambodian': 195, 'Campgrounds': 196, 'Canadian (New)': 197, 'Candle Stores': 198, 'Candy Stores': 199, 'Cannabis Clinics': 200, 'Cannabis Collective': 201, 'Cannabis Dispensaries': 202, 'Cannabis Tours': 203, 'Cantonese': 204, 'Car Auctions': 205, 'Car Brokers': 206, 'Car Buyers': 207, 'Car Dealers': 208, 'Car Inspectors': 209, 'Car Rental': 210, 'Car Share Services': 211, 'Car Stereo Installation': 212, 'Car Wash': 213, 'Car Window Tinting': 214, 'Cardio Classes': 215, 'Cardiologists': 216, 'Cards & Stationery': 217, 'Career Counseling': 218, 'Caribbean': 219, 'Caricatures': 220, 'Carousels': 221, 'Carpenters': 222, 'Carpet Cleaning': 223, 'Carpet Dyeing': 224, 'Carpet Installation': 225, 'Carpeting': 226, 'Casinos': 227, 'Castles': 228, 'Caterers': 229, 'Ceremonial Clothing': 230, 'Challenge Courses': 231, 'Champagne Bars': 232, 'Check Cashing/Pay-day Loans': 233, 'Cheerleading': 234, 'Cheese Shops': 235, 'Cheese Tasting Classes': 236, 'Cheesesteaks': 237, 'Chicken Shop': 238, 'Chicken Wings': 239, 'Child Care & Day Care': 240, 'Childbirth Education': 241, 'Childproofing': 242, "Children's Clothing": 243, "Children's Museums": 244, 'Chilean': 245, 'Chimney Sweeps': 246, 'Chinese': 247, 'Chinese Martial Arts': 248, 'Chiropractors': 249, 'Chocolatiers & Shops': 250, 'Christmas Markets': 251, 'Christmas Trees': 252, 'Churches': 253, 'Churros': 254, 'Cideries': 255, 'Cigar Bars': 256, 'Cinema': 257, 'Circuit Training Gyms': 258, 'Civic Center': 259, 'Climbing': 260, 'Clock Repair': 261, 'Clothing Rental': 262, 'Clowns': 263, 'Club Crawl': 264, 'Cocktail Bars': 265, 'Coffee & Tea': 266, 'Coffee & Tea Supplies': 267, 'Coffee Roasteries': 268, 'Coffeeshops': 269, 'College Counseling': 270, 'Colleges & Universities': 271, 'Colombian': 272, 'Colonics': 273, 'Comedy Clubs': 274, 'Comfort Food': 275, 'Comic Books': 276, 'Commercial Real Estate': 277, 'Commercial Truck Dealers': 278, 'Commercial Truck Repair': 279, 'Commissioned Artists': 280, 'Community Centers': 281, 'Community Gardens': 282, 'Community Service/Non-Profit': 283, 'Computers': 284, 'Concept Shops': 285, 'Concierge Medicine': 286, 'Condominiums': 287, 'Contract Law': 288, 'Contractors': 289, 'Convenience Stores': 290, 'Conveyor Belt Sushi': 291, 'Cooking Classes': 292, 'Cooking Schools': 293, 'Cosmetic Dentists': 294, 'Cosmetic Surgeons': 295, 'Cosmetics & Beauty Supply': 296, 'Cosmetology Schools': 297, 'Costumes': 298, 'Counseling & Mental Health': 299, 'Countertop Installation': 300, 'Country Clubs': 301, 'Country Dance Halls': 302, 'Couriers & Delivery Services': 303, 'Court Reporters': 304, 'Courthouses': 305, 'Crane Services': 306, 'Cremation Services': 307, 'Creperies': 308, 'Criminal Defense Law': 309, 'Cryotherapy': 310, 'Cuban': 311, 'Cultural Center': 312, 'Cupcakes': 313, 'Currency Exchange': 314, 'Custom Cakes': 315, 'Customized Merchandise': 316, 'Customs Brokers': 317, 'Cycling Classes': 318, 'Czech': 319, 'Czech/Slovakian': 320, 'DIY Auto Shop': 321, 'DJs': 322, 'DUI Law': 323, 'DUI Schools': 324, 'Damage Restoration': 325, 'Dance Clubs': 326, 'Dance Schools': 327, 'Dance Studios': 328, 'Dance Wear': 329, 'Data Recovery': 330, 'Day Camps': 331, 'Day Spas': 332, 'Debt Relief Services': 333, 'Decks & Railing': 334, 'Delicatessen': 335, 'Delis': 336, 'Demolition Services': 337, 'Dental Hygienists': 338, 'Dentists': 339, 'Department Stores': 340, 'Departments of Motor Vehicles': 341, 'Dermatologists': 342, 'Desserts': 343, 'Diagnostic Imaging': 344, 'Diagnostic Services': 345, 'Dialysis Clinics': 346, 'Dietitians': 347, 'Digitizing Services': 348, 'Dim Sum': 349, 'Diners': 350, 'Dinner Theater': 351, 'Disability Law': 352, 'Disc Golf': 353, 'Discount Store': 354, 'Distilleries': 355, 'Dive Bars': 356, 'Dive Shops': 357, 'Diving': 358, 'Divorce & Family Law': 359, 'Do-It-Yourself Food': 360, 'Doctors': 361, 'Dog Parks': 362, 'Dog Walkers': 363, 'Dominican': 364, 'Donairs': 365, 'Donation Center': 366, 'Donuts': 367, 'Door Sales/Installation': 368, 'Doulas': 369, 'Drive-In Theater': 370, 'Drive-Thru Bars': 371, 'Driving Schools': 372, 'Drones': 373, 'Drugstores': 374, 'Dry Cleaning': 375, 'Dry Cleaning & Laundry': 376, 'Drywall Installation & Repair': 377, 'Dumpster Rental': 378, 'Duplication Services': 379, 'Duty-Free Shops': 380, 'EV Charging Stations': 381, 'Ear Nose & Throat': 382, 'Eastern European': 383, 'Eatertainment': 384, 'Editorial Services': 385, 'Education': 386, 'Educational Services': 387, 'Egyptian': 388, 'Elder Care Planning': 389, 'Electricians': 390, 'Electricity Suppliers': 391, 'Electronics': 392, 'Electronics Repair': 393, 'Elementary Schools': 394, 'Embassy': 395, 'Embroidery & Crochet': 396, 'Emergency Medicine': 397, 'Emergency Pet Hospital': 398, 'Emergency Rooms': 399, 'Empanadas': 400, 'Employment Agencies': 401, 'Employment Law': 402, 'Endocrinologists': 403, 'Endodontists': 404, 'Engraving': 405, 'Entertainment Law': 406, 'Environmental Abatement': 407, 'Environmental Testing': 408, 'Erotic Massage': 409, 'Escape Games': 410, 'Estate Liquidation': 411, 'Estate Planning Law': 412, 'Estheticians': 413, 'Ethical Grocery': 414, 'Ethiopian': 415, 'Ethnic Food': 416, 'Ethnic Grocery': 417, 'Event Photography': 418, 'Event Planning & Services': 419, 'Excavation Services': 420, 'Experiences': 421, 'Eyebrow Services': 422, 'Eyelash Service': 423, 'Eyewear & Opticians': 424, 'Fabric Stores': 425, 'Face Painting': 426, 'Falafel': 427, 'Family Practice': 428, 'Farm Equipment Repair': 429, 'Farmers Market': 430, 'Farming Equipment': 431, 'Farms': 432, 'Fashion': 433, 'Fast Food': 434, 'Fences & Gates': 435, 'Fencing Clubs': 436, 'Feng Shui': 437, 'Fertility': 438, 'Festivals': 439, 'Filipino': 440, 'Financial Advising': 441, 'Financial Services': 442, 'Fingerprinting': 443, 'Fire Departments': 444, 'Fire Protection Services': 445, 'Firearm Training': 446, 'Fireplace Services': 447, 'Firewood': 448, 'Fireworks': 449, 'First Aid Classes': 450, 'Fish & Chips': 451, 'Fishing': 452, 'Fishmonger': 453, 'Fitness & Instruction': 454, 'Fitness/Exercise Equipment': 455, 'Flea Markets': 456, 'Flight Instruction': 457, 'Float Spa': 458, 'Flooring': 459, 'Floral Designers': 460, 'Florists': 461, 'Flowers': 462, 'Flowers & Gifts': 463, 'Flyboarding': 464, 'Fondue': 465, 'Food': 466, 'Food Banks': 467, 'Food Court': 468, 'Food Delivery Services': 469, 'Food Stands': 470, 'Food Tours': 471, 'Food Trucks': 472, 'Formal Wear': 473, 'Foundation Repair': 474, 'Framing': 475, 'Free Diving': 476, 'French': 477, 'Fruits & Veggies': 478, 'Fuel Docks': 479, 'Funeral Services & Cemeteries': 480, 'Fur Clothing': 481, 'Furniture Assembly': 482, 'Furniture Rental': 483, 'Furniture Repair': 484, 'Furniture Reupholstery': 485, 'Furniture Stores': 486, 'Game Meat': 487, 'Game Truck Rental': 488, 'Garage Door Services': 489, 'Gardeners': 490, 'Gas Stations': 491, 'Gastroenterologist': 492, 'Gastropubs': 493, 'Gay Bars': 494, 'Gelato': 495, 'Gemstones & Minerals': 496, 'General Dentistry': 497, 'General Festivals': 498, 'General Litigation': 499, 'Generator Installation/Repair': 500, 'German': 501, 'Gerontologists': 502, 'Gift Shops': 503, 'Glass & Mirrors': 504, 'Glass Blowing': 505, 'Gluten-Free': 506, 'Go Karts': 507, 'Gold Buyers': 508, 'Golf': 509, 'Golf Cart Dealers': 510, 'Golf Cart Rentals': 511, 'Golf Equipment': 512, 'Golf Equipment Shops': 513, 'Golf Lessons': 514, 'Graphic Design': 515, 'Greek': 516, 'Grilling Equipment': 517, 'Grocery': 518, 'Grout Services': 519, 'Guamanian': 520, 'Guest Houses': 521, 'Guitar Stores': 522, 'Gun/Rifle Ranges': 523, 'Guns & Ammo': 524, 'Gunsmith': 525, 'Gutter Services': 526, 'Gymnastics': 527, 'Gyms': 528, 'Habilitative Services': 529, 'Hainan': 530, 'Hair Extensions': 531, 'Hair Loss Centers': 532, 'Hair Removal': 533, 'Hair Salons': 534, 'Hair Stylists': 535, 'Haitian': 536, 'Hakka': 537, 'Halal': 538, 'Halfway Houses': 539, 'Halotherapy': 540, 'Handyman': 541, 'Hang Gliding': 542, 'Hardware Stores': 543, 'Hats': 544, 'Haunted Houses': 545, 'Hawaiian': 546, 'Hazardous Waste Disposal': 547, 'Head Shops': 548, 'Health & Medical': 549, 'Health Coach': 550, 'Health Insurance Offices': 551, 'Health Markets': 552, 'Health Retreats': 553, 'Hearing Aid Providers': 554, 'Hearing Aids': 555, 'Heating & Air Conditioning/HVAC': 556, 'Henna Artists': 557, 'Hepatologists': 558, 'Herbal Shops': 559, 'Herbs & Spices': 560, 'High Fidelity Audio Equipment': 561, 'Hiking': 562, 'Himalayan/Nepalese': 563, 'Hindu Temples': 564, 'Historical Tours': 565, 'Hobby Shops': 566, 'Hockey Equipment': 567, 'Holiday Decorating Services': 568, 'Holiday Decorations': 569, 'Holistic Animal Care': 570, 'Home & Garden': 571, 'Home & Rental Insurance': 572, 'Home Automation': 573, 'Home Cleaning': 574, 'Home Decor': 575, 'Home Developers': 576, 'Home Energy Auditors': 577, 'Home Health Care': 578, 'Home Inspectors': 579, 'Home Network Installation': 580, 'Home Organization': 581, 'Home Services': 582, 'Home Staging': 583, 'Home Theatre Installation': 584, 'Home Window Tinting': 585, 'Homeless Shelters': 586, 'Homeowner Association': 587, 'Honduran': 588, 'Honey': 589, 'Hong Kong Style Cafe': 590, 'Hookah Bars': 591, 'Horse Boarding': 592, 'Horse Equipment Shops': 593, 'Horse Racing': 594, 'Horseback Riding': 595, 'Hospice': 596, 'Hospitals': 597, 'Hostels': 598, 'Hot Air Balloons': 599, 'Hot Dogs': 600, 'Hot Pot': 601, 'Hot Tub & Pool': 602, 'Hotel bar': 603, 'Hotels': 604, 'Hotels & Travel': 605, 'House Sitters': 606, 'Hungarian': 607, 'Hunting & Fishing Supplies': 608, 'Hybrid Car Repair': 609, 'Hydro-jetting': 610, 'Hydroponics': 611, 'Hydrotherapy': 612, 'Hypnosis/Hypnotherapy': 613, 'IP & Internet Law': 614, 'IT Services & Computer Repair': 615, 'IV Hydration': 616, 'Iberian': 617, 'Ice Cream & Frozen Yogurt': 618, 'Ice Delivery': 619, 'Immigration Law': 620, 'Immunodermatologists': 621, 'Imported Food': 622, 'Indian': 623, 'Indonesian': 624, 'Indoor Landscaping': 625, 'Indoor Playcentre': 626, 'Infectious Disease Specialists': 627, 'Installment Loans': 628, 'Insulation Installation': 629, 'Insurance': 630, 'Interior Design': 631, 'Interlock Systems': 632, 'Internal Medicine': 633, 'International': 634, 'International Grocery': 635, 'Internet Cafes': 636, 'Internet Service Providers': 637, 'Interval Training Gyms': 638, 'Investing': 639, 'Irish': 640, 'Irish Pub': 641, 'Irrigation': 642, 'Island Pub': 643, 'Italian': 644, 'Izakaya': 645, 'Jails & Prisons': 646, 'Japanese': 647, 'Japanese Curry': 648, 'Japanese Sweets': 649, 'Jazz & Blues': 650, 'Jet Skis': 651, 'Jewelry': 652, 'Jewelry Repair': 653, 'Juice Bars & Smoothies': 654, 'Junk Removal & Hauling': 655, 'Junkyards': 656, 'Karaoke': 657, 'Karate': 658, 'Kebab': 659, 'Keys & Locksmiths': 660, 'Kickboxing': 661, 'Kids Activities': 662, 'Kids Hair Salons': 663, 'Kitchen & Bath': 664, 'Kitchen Incubators': 665, 'Kitchen Supplies': 666, 'Knife Sharpening': 667, 'Knitting Supplies': 668, 'Kombucha': 669, 'Korean': 670, 'Kosher': 671, 'LAN Centers': 672, 'Laboratory Testing': 673, 'Lactation Services': 674, 'Lakes': 675, 'Land Surveying': 676, 'Landmarks & Historical Buildings': 677, 'Landscape Architects': 678, 'Landscaping': 679, 'Language Schools': 680, 'Laotian': 681, 'Laser Eye Surgery/Lasik': 682, 'Laser Hair Removal': 683, 'Laser Tag': 684, 'Latin American': 685, 'Laundromat': 686, 'Laundry Services': 687, 'Lawn Services': 688, 'Lawyers': 689, 'Leather Goods': 690, 'Lebanese': 691, 'Legal Services': 692, 'Leisure Centers': 693, 'Libraries': 694, 'Lice Services': 695, 'Life Coach': 696, 'Life Insurance': 697, 'Lighting Fixtures & Equipment': 698, 'Lighting Stores': 699, 'Limos': 700, 'Lingerie': 701, 'Live/Raw Food': 702, 'Livestock Feed & Supply': 703, 'Local Fish Stores': 704, 'Local Flavor': 705, 'Local Services': 706, 'Lounges': 707, 'Luggage': 708, 'Luggage Storage': 709, 'Macarons': 710, 'Machine & Tool Rental': 711, 'Machine Shops': 712, 'Magicians': 713, 'Mags': 714, 'Mailbox Centers': 715, 'Makerspaces': 716, 'Makeup Artists': 717, 'Malaysian': 718, 'Marinas': 719, 'Market Stalls': 720, 'Marketing': 721, 'Martial Arts': 722, 'Masonry/Concrete': 723, 'Mass Media': 724, 'Massage': 725, 'Massage Schools': 726, 'Massage Therapy': 727, 'Matchmakers': 728, 'Maternity Wear': 729, 'Mattresses': 730, 'Mauritius': 731, 'Meat Shops': 732, 'Mediators': 733, 'Medical Cannabis Referrals': 734, 'Medical Centers': 735, 'Medical Foot Care': 736, 'Medical Law': 737, 'Medical Spas': 738, 'Medical Supplies': 739, 'Medical Transportation': 740, 'Meditation Centers': 741, 'Mediterranean': 742, 'Memory Care': 743, "Men's Clothing": 744, "Men's Hair Salons": 745, 'Metal Fabricators': 746, 'Metro Stations': 747, 'Mexican': 748, 'Middle Eastern': 749, 'Middle Schools & High Schools': 750, 'Midwives': 751, 'Military Surplus': 752, 'Milkshake Bars': 753, 'Minho': 754, 'Mini Golf': 755, 'Misting System Services': 756, 'Mobile Dent Repair': 757, 'Mobile Home Dealers': 758, 'Mobile Home Parks': 759, 'Mobile Home Repair': 760, 'Mobile Phone Accessories': 761, 'Mobile Phone Repair': 762, 'Mobile Phones': 763, 'Mobility Equipment Sales & Services': 764, 'Modern European': 765, 'Mongolian': 766, 'Montessori Schools': 767, 'Moroccan': 768, 'Mortgage Brokers': 769, 'Mortgage Lenders': 770, 'Mortuary Services': 771, 'Mosques': 772, 'Motorcycle Dealers': 773, 'Motorcycle Gear': 774, 'Motorcycle Rental': 775, 'Motorcycle Repair': 776, 'Motorsport Vehicle Dealers': 777, 'Motorsport Vehicle Repairs': 778, 'Mountain Biking': 779, 'Movers': 780, 'Muay Thai': 781, 'Municipality': 782, 'Museums': 783, 'Music & DVDs': 784, 'Music & Video': 785, 'Music Production Services': 786, 'Music Venues': 787, 'Musical Instrument Services': 788, 'Musical Instruments & Teachers': 789, 'Musicians': 790, 'Nail Salons': 791, 'Nail Technicians': 792, 'Nanny Services': 793, 'Natural Gas Suppliers': 794, 'Naturopathic/Holistic': 795, 'Nephrologists': 796, 'Neurologist': 797, 'Neuropathologists': 798, 'Neurotologists': 799, 'New Mexican Cuisine': 800, 'Newspapers & Magazines': 801, 'Nicaraguan': 802, 'Nightlife': 803, 'Noodles': 804, 'Northern German': 805, 'Notaries': 806, 'Nudist': 807, 'Nurse Practitioner': 808, 'Nurseries & Gardening': 809, 'Nursing Schools': 810, 'Nutritionists': 811, 'Observatories': 812, 'Obstetricians & Gynecologists': 813, 'Occupational Therapy': 814, 'Office Cleaning': 815, 'Office Equipment': 816, 'Officiants': 817, 'Oil Change Stations': 818, 'Olive Oil': 819, 'Oncologist': 820, 'Opera & Ballet': 821, 'Ophthalmologists': 822, 'Optometrists': 823, 'Oral Surgeons': 824, 'Organic Stores': 825, 'Orthodontists': 826, 'Orthopedists': 827, 'Orthotics': 828, 'Osteopathic Physicians': 829, 'Osteopaths': 830, 'Otologists': 831, 'Outdoor Furniture Stores': 832, 'Outdoor Gear': 833, 'Outdoor Movies': 834, 'Outlet Stores': 835, 'Oxygen Bars': 836, 'Packing Services': 837, 'Packing Supplies': 838, 'Paddleboarding': 839, 'Pain Management': 840, 'Paint & Sip': 841, 'Paint Stores': 842, 'Paint-Your-Own Pottery': 843, 'Paintball': 844, 'Painters': 845, 'Pakistani': 846, 'Palatine': 847, 'Pan Asian': 848, 'Parenting Classes': 849, 'Parking': 850, 'Parks': 851, 'Party & Event Planning': 852, 'Party Bike Rentals': 853, 'Party Bus Rentals': 854, 'Party Characters': 855, 'Party Equipment Rentals': 856, 'Party Supplies': 857, 'Passport & Visa Services': 858, 'Pasta Shops': 859, 'Patent Law': 860, 'Pathologists': 861, 'Patio Coverings': 862, 'Patisserie/Cake Shop': 863, 'Pawn Shops': 864, 'Payroll Services': 865, 'Pediatric Dentists': 866, 'Pediatricians': 867, 'Pedicabs': 868, 'Pensions': 869, 'Performing Arts': 870, 'Perfume': 871, 'Periodontists': 872, 'Permanent Makeup': 873, 'Persian/Iranian': 874, 'Personal Assistants': 875, 'Personal Care Services': 876, 'Personal Chefs': 877, 'Personal Injury Law': 878, 'Personal Shopping': 879, 'Peruvian': 880, 'Pest Control': 881, 'Pet Adoption': 882, 'Pet Boarding': 883, 'Pet Breeders': 884, 'Pet Cremation Services': 885, 'Pet Groomers': 886, 'Pet Hospice': 887, 'Pet Insurance': 888, 'Pet Photography': 889, 'Pet Services': 890, 'Pet Sitting': 891, 'Pet Stores': 892, 'Pet Training': 893, 'Pet Transportation': 894, 'Pet Waste Removal': 895, 'Pets': 896, 'Petting Zoos': 897, 'Pharmacy': 898, 'Phlebologists': 899, 'Photo Booth Rentals': 900, 'Photographers': 901, 'Photography Classes': 902, 'Photography Stores & Services': 903, 'Physical Therapy': 904, 'Piano Bars': 905, 'Piano Services': 906, 'Piano Stores': 907, 'Pick Your Own Farms': 908, 'Piercing': 909, 'Pilates': 910, 'Pita': 911, 'Pizza': 912, 'Placenta Encapsulations': 913, 'Planetarium': 914, 'Plastic Surgeons': 915, 'Playgrounds': 916, 'Playsets': 917, 'Plumbing': 918, 'Plus Size Fashion': 919, 'Podiatrists': 920, 'Poke': 921, 'Pole Dancing Classes': 922, 'Police Departments': 923, 'Polish': 924, 'Pool & Billiards': 925, 'Pool & Hot Tub Service': 926, 'Pool Cleaners': 927, 'Pool Halls': 928, 'Pop-Up Restaurants': 929, 'Pop-up Shops': 930, 'Popcorn Shops': 931, 'Portuguese': 932, 'Post Offices': 933, 'Poutineries': 934, 'Powder Coating': 935, 'Prenatal/Perinatal Care': 936, 'Preschools': 937, 'Pressure Washers': 938, 'Pretzels': 939, 'Preventive Medicine': 940, 'Print Media': 941, 'Printing Services': 942, 'Private Investigation': 943, 'Private Jet Charter': 944, 'Private Schools': 945, 'Private Tutors': 946, 'Process Servers': 947, 'Proctologists': 948, 'Product Design': 949, 'Professional Services': 950, 'Professional Sports Teams': 951, 'Propane': 952, 'Property Management': 953, 'Props': 954, 'Prosthetics': 955, 'Prosthodontists': 956, 'Psychiatrists': 957, 'Psychic Mediums': 958, 'Psychics': 959, 'Psychologists': 960, 'Pub Food': 961, 'Public Adjusters': 962, 'Public Art': 963, 'Public Markets': 964, 'Public Relations': 965, 'Public Services & Government': 966, 'Public Transportation': 967, 'Pubs': 968, 'Puerto Rican': 969, 'Pulmonologist': 970, 'Pumpkin Patches': 971, 'Qi Gong': 972, 'RV Dealers': 973, 'RV Parks': 974, 'RV Rental': 975, 'RV Repair': 976, 'Race Tracks': 977, 'Races & Competitions': 978, 'Racing Experience': 979, 'Radio Stations': 980, 'Radiologists': 981, 'Rafting/Kayaking': 982, 'Ramen': 983, 'Ranches': 984, 'Real Estate': 985, 'Real Estate Agents': 986, 'Real Estate Law': 987, 'Real Estate Photography': 988, 'Real Estate Services': 989, 'Recording & Rehearsal Studios': 990, 'Recreation Centers': 991, 'Recycling Center': 992, 'Refinishing Services': 993, 'Reflexology': 994, 'Registration Services': 995, 'Registry Office': 996, 'Rehabilitation Center': 997, 'Reiki': 998, 'Religious Items': 999, 'Religious Organizations': 1000, 'Religious Schools': 1001, 'Reptile Shops': 1002, 'Resorts': 1003, 'Rest Stops': 1004, 'Restaurant Supplies': 1005, 'Restaurants': 1006, 'Retina Specialists': 1007, 'Retirement Homes': 1008, 'Reunion': 1009, 'Rheumatologists': 1010, 'Roadside Assistance': 1011, 'Rock Climbing': 1012, 'Rodeo': 1013, 'Rolfing': 1014, 'Roof Inspectors': 1015, 'Roofing': 1016, 'Rotisserie Chicken': 1017, 'Rugs': 1018, 'Russian': 1019, 'Safe Stores': 1020, 'Safety Equipment': 1021, 'Sailing': 1022, 'Salad': 1023, 'Salvadoran': 1024, 'Sandblasting': 1025, 'Sandwiches': 1026, 'Sauna Installation & Repair': 1027, 'Saunas': 1028, 'Scandinavian': 1029, 'Scavenger Hunts': 1030, 'Scooter Rentals': 1031, 'Scooter Tours': 1032, 'Scottish': 1033, 'Screen Printing': 1034, 'Screen Printing/T-Shirt Printing': 1035, 'Scuba Diving': 1036, 'Seafood': 1037, 'Seafood Markets': 1038, 'Security Services': 1039, 'Security Systems': 1040, 'Self Storage': 1041, 'Self-defense Classes': 1042, 'Senegalese': 1043, 'Senior Centers': 1044, 'Septic Services': 1045, 'Serbo Croatian': 1046, 'Service Stations': 1047, 'Session Photography': 1048, 'Sewing & Alterations': 1049, 'Sex Therapists': 1050, 'Shades & Blinds': 1051, 'Shanghainese': 1052, 'Shared Office Spaces': 1053, 'Shaved Ice': 1054, 'Shaved Snow': 1055, 'Shipping Centers': 1056, 'Shoe Repair': 1057, 'Shoe Shine': 1058, 'Shoe Stores': 1059, 'Shopping': 1060, 'Shopping Centers': 1061, 'Shredding Services': 1062, 'Shutters': 1063, 'Sicilian': 1064, 'Siding': 1065, 'Signature Cuisine': 1066, 'Signmaking': 1067, 'Singaporean': 1068, 'Skate Parks': 1069, 'Skate Shops': 1070, 'Skating Rinks': 1071, 'Ski & Snowboard Shops': 1072, 'Ski Resorts': 1073, 'Ski Schools': 1074, 'Skiing': 1075, 'Skilled Nursing': 1076, 'Skin Care': 1077, 'Skydiving': 1078, 'Sledding': 1079, 'Sleep Specialists': 1080, 'Slovakian': 1081, 'Smog Check Stations': 1082, 'Smokehouse': 1083, 'Snorkeling': 1084, 'Snow Removal': 1085, 'Soba': 1086, 'Soccer': 1087, 'Social Clubs': 1088, 'Social Security Law': 1089, 'Software Development': 1090, 'Solar Installation': 1091, 'Solar Panel Cleaning': 1092, 'Soul Food': 1093, 'Soup': 1094, 'South African': 1095, 'Southern': 1096, 'Souvenir Shops': 1097, 'Spanish': 1098, 'Speakeasies': 1099, 'Special Education': 1100, 'Specialty Food': 1101, 'Specialty Schools': 1102, 'Speech Therapists': 1103, 'Spin Classes': 1104, 'Spine Surgeons': 1105, 'Spiritual Shop': 1106, 'Sport Equipment Hire': 1107, 'Sporting Goods': 1108, 'Sports Bars': 1109, 'Sports Clubs': 1110, 'Sports Medicine': 1111, 'Sports Psychologists': 1112, 'Sports Wear': 1113, 'Spray Tanning': 1114, 'Squash': 1115, 'Sri Lankan': 1116, 'Stadiums & Arenas': 1117, 'Steakhouses': 1118, 'Storefront Clinics': 1119, 'Street Art': 1120, 'Street Vendors': 1121, 'Strip Clubs': 1122, 'Striptease Dancers': 1123, 'Structural Engineers': 1124, 'Stucco Services': 1125, 'Sugar Shacks': 1126, 'Sugaring': 1127, 'Summer Camps': 1128, 'Sunglasses': 1129, 'Supernatural Readings': 1130, 'Supper Clubs': 1131, 'Surf Schools': 1132, 'Surf Shop': 1133, 'Surfing': 1134, 'Surgeons': 1135, 'Sushi Bars': 1136, 'Swimming Lessons/Schools': 1137, 'Swimming Pools': 1138, 'Swimwear': 1139, 'Swiss Food': 1140, 'Synagogues': 1141, 'Syrian': 1142, 'Szechuan': 1143, 'TV Mounting': 1144, 'Tabletop Games': 1145, 'Tableware': 1146, 'Tacos': 1147, 'Taekwondo': 1148, 'Tai Chi': 1149, 'Taiwanese': 1150, 'Talent Agencies': 1151, 'Tanning': 1152, 'Tanning Beds': 1153, 'Tapas Bars': 1154, 'Tapas/Small Plates': 1155, 'Tasting Classes': 1156, 'Tattoo': 1157, 'Tattoo Removal': 1158, 'Tax Law': 1159, 'Tax Services': 1160, 'Taxidermy': 1161, 'Taxis': 1162, 'Tea Rooms': 1163, 'Teacher Supplies': 1164, 'Team Building Activities': 1165, 'Teeth Whitening': 1166, 'Telecommunications': 1167, 'Television Service Providers': 1168, 'Television Stations': 1169, 'Tempura': 1170, 'Tenant and Eviction Law': 1171, 'Tennis': 1172, 'Teppanyaki': 1173, 'Test Preparation': 1174, 'Tex-Mex': 1175, 'Thai': 1176, 'Themed Cafes': 1177, 'Threading Services': 1178, 'Thrift Stores': 1179, 'Ticket Sales': 1180, 'Tickets': 1181, 'Tiki Bars': 1182, 'Tiling': 1183, 'Tires': 1184, 'Title Loans': 1185, 'Tobacco Shops': 1186, 'Tonkatsu': 1187, 'Tours': 1188, 'Towing': 1189, 'Town Car Service': 1190, 'Town Hall': 1191, 'Toxicologists': 1192, 'Toy Stores': 1193, 'Traditional Chinese Medicine': 1194, 'Traditional Clothing': 1195, 'Traditional Norwegian': 1196, 'Traffic Schools': 1197, 'Traffic Ticketing Law': 1198, 'Trailer Dealers': 1199, 'Trailer Rental': 1200, 'Trailer Repair': 1201, 'Train Stations': 1202, 'Trainers': 1203, 'Trains': 1204, 'Trampoline Parks': 1205, 'Translation Services': 1206, 'Transmission Repair': 1207, 'Transportation': 1208, 'Travel Agents': 1209, 'Travel Services': 1210, 'Tree Services': 1211, 'Trinidadian': 1212, 'Trivia Hosts': 1213, 'Trophy Shops': 1214, 'Truck Rental': 1215, 'Trusts': 1216, 'Tubing': 1217, 'Tui Na': 1218, 'Turkish': 1219, 'Tuscan': 1220, 'Tutoring Centers': 1221, 'Udon': 1222, 'Ukrainian': 1223, 'Ultrasound Imaging Centers': 1224, 'Undersea/Hyperbaric Medicine': 1225, 'Uniforms': 1226, 'University Housing': 1227, 'Unofficial Yelp Events': 1228, 'Urgent Care': 1229, 'Urologists': 1230, 'Used': 1231, 'Used Bookstore': 1232, 'Used Car Dealers': 1233, 'Utilities': 1234, 'Uzbek': 1235, 'Vacation Rental Agents': 1236, 'Vacation Rentals': 1237, 'Valet Services': 1238, 'Vape Shops': 1239, 'Vascular Medicine': 1240, 'Vegan': 1241, 'Vegetarian': 1242, 'Vehicle Shipping': 1243, 'Vehicle Wraps': 1244, 'Venezuelan': 1245, 'Venues & Event Spaces': 1246, 'Veterans Organizations': 1247, 'Veterinarians': 1248, 'Video Game Stores': 1249, 'Video/Film Production': 1250, 'Videographers': 1251, 'Videos & Video Game Rental': 1252, 'Vietnamese': 1253, 'Vintage & Consignment': 1254, 'Vinyl Records': 1255, 'Virtual Reality Centers': 1256, 'Visitor Centers': 1257, 'Vitamins & Supplements': 1258, 'Vocal Coach': 1259, 'Vocational & Technical School': 1260, 'Waffles': 1261, 'Waldorf Schools': 1262, 'Walk-in Clinics': 1263, 'Walking Tours': 1264, 'Wallpapering': 1265, 'Watch Repair': 1266, 'Watches': 1267, 'Water Delivery': 1268, 'Water Heater Installation/Repair': 1269, 'Water Parks': 1270, 'Water Purification Services': 1271, 'Water Stores': 1272, 'Water Suppliers': 1273, 'Waterproofing': 1274, 'Waxing': 1275, 'Web Design': 1276, 'Wedding Chapels': 1277, 'Wedding Planning': 1278, 'Weight Loss Centers': 1279, 'Well Drilling': 1280, 'Wheel & Rim Repair': 1281, 'Whiskey Bars': 1282, 'Wholesale Stores': 1283, 'Wholesalers': 1284, 'Wigs': 1285, 'Wildlife Control': 1286, 'Wildlife Hunting Ranges': 1287, 'Wills': 1288, 'Window Washing': 1289, 'Windows Installation': 1290, 'Windshield Installation & Repair': 1291, 'Wine & Spirits': 1292, 'Wine Bars': 1293, 'Wine Tasting Classes': 1294, 'Wine Tasting Room': 1295, 'Wine Tours': 1296, 'Wineries': 1297, "Women's Clothing": 1298, 'Workers Compensation Law': 1299, 'Wraps': 1300, 'Yelp Events': 1301, 'Yoga': 1302, 'Ziplining': 1303, 'Zoos': 1304, 'attributes$AcceptsInsurance': 1305, 'attributes$AgesAllowed': 1306, 'attributes$Alcohol': 1307, 'attributes$Ambience$casual': 1308, 'attributes$Ambience$classy': 1309, 'attributes$Ambience$divey': 1310, 'attributes$Ambience$hipster': 1311, 'attributes$Ambience$intimate': 1312, 'attributes$Ambience$romantic': 1313, 'attributes$Ambience$touristy': 1314, 'attributes$Ambience$trendy': 1315, 'attributes$Ambience$upscale': 1316, 'attributes$BYOB': 1317, 'attributes$BYOBCorkage': 1318, 'attributes$BestNights$friday': 1319, 'attributes$BestNights$monday': 1320, 'attributes$BestNights$saturday': 1321, 'attributes$BestNights$sunday': 1322, 'attributes$BestNights$thursday': 1323, 'attributes$BestNights$tuesday': 1324, 'attributes$BestNights$wednesday': 1325, 'attributes$BikeParking': 1326, 'attributes$BusinessAcceptsBitcoin': 1327, 'attributes$BusinessAcceptsCreditCards': 1328, 'attributes$BusinessParking$garage': 1329, 'attributes$BusinessParking$lot': 1330, 'attributes$BusinessParking$street': 1331, 'attributes$BusinessParking$valet': 1332, 'attributes$BusinessParking$validated': 1333, 'attributes$ByAppointmentOnly': 1334, 'attributes$Caters': 1335, 'attributes$CoatCheck': 1336, 'attributes$Corkage': 1337, 'attributes$DietaryRestrictions$dairy-free': 1338, 'attributes$DietaryRestrictions$gluten-free': 1339, 'attributes$DietaryRestrictions$halal': 1340, 'attributes$DietaryRestrictions$kosher': 1341, 'attributes$DietaryRestrictions$soy-free': 1342, 'attributes$DietaryRestrictions$vegan': 1343, 'attributes$DietaryRestrictions$vegetarian': 1344, 'attributes$DogsAllowed': 1345, 'attributes$DriveThru': 1346, 'attributes$GoodForDancing': 1347, 'attributes$GoodForKids': 1348, 'attributes$GoodForMeal$breakfast': 1349, 'attributes$GoodForMeal$brunch': 1350, 'attributes$GoodForMeal$dessert': 1351, 'attributes$GoodForMeal$dinner': 1352, 'attributes$GoodForMeal$latenight': 1353, 'attributes$GoodForMeal$lunch': 1354, 'attributes$HairSpecializesIn$africanamerican': 1355, 'attributes$HairSpecializesIn$asian': 1356, 'attributes$HairSpecializesIn$coloring': 1357, 'attributes$HairSpecializesIn$curly': 1358, 'attributes$HairSpecializesIn$extensions': 1359, 'attributes$HairSpecializesIn$kids': 1360, 'attributes$HairSpecializesIn$perms': 1361, 'attributes$HairSpecializesIn$straightperms': 1362, 'attributes$HappyHour': 1363, 'attributes$HasTV': 1364, 'attributes$Music$background_music': 1365, 'attributes$Music$dj': 1366, 'attributes$Music$jukebox': 1367, 'attributes$Music$karaoke': 1368, 'attributes$Music$live': 1369, 'attributes$Music$no_music': 1370, 'attributes$Music$video': 1371, 'attributes$NoiseLevel': 1372, 'attributes$Open24Hours': 1373, 'attributes$OutdoorSeating': 1374, 'attributes$RestaurantsAttire': 1375, 'attributes$RestaurantsCounterService': 1376, 'attributes$RestaurantsDelivery': 1377, 'attributes$RestaurantsGoodForGroups': 1378, 'attributes$RestaurantsPriceRange2': 1379, 'attributes$RestaurantsReservations': 1380, 'attributes$RestaurantsTableService': 1381, 'attributes$RestaurantsTakeOut': 1382, 'attributes$Smoking': 1383, 'attributes$WheelchairAccessible': 1384, 'attributes$WiFi': 1385, 'hours$Friday': 1386, 'hours$Monday': 1387, 'hours$Saturday': 1388, 'hours$Sunday': 1389, 'hours$Thursday': 1390, 'hours$Tuesday': 1391, 'hours$Wednesday': 1392, 'is_open': 1393, 'latitude': 1394, 'longitude': 1395, 'review_count': 1396, 'stars': 1397, 'state': 1398}

#----------- Functions for feature extraction
def convert_timings_to_hours(timing):
    start_time, end_time = timing.split('-')
    start_time = datetime.strptime(start_time, '%H:%M')
    end_time = datetime.strptime(end_time, '%H:%M')
    time_diff = end_time - start_time
    num_hours = time_diff.total_seconds() / 3600

    if num_hours < 0:
        num_hours = 24+num_hours

    return num_hours

def get_latitude(latitude_value):
    if not latitude_value:
        return 0
    return latitude_value

def get_longitude(longitude_value):
    if not longitude_value:
        return 0
    return longitude_value

def get_num_attributes(attributes_dict):
    if not attributes_dict:
        return 0
    return len(attributes_dict)

def get_rate_true_attributes(attributes_dict):
    if not attributes_dict:
        return 0
    num_total = 0
    num_true = 0
    for k,v in attributes_dict.items():
        if v in ('True', 'False'):
            num_total += 1
            if v == 'True':
                num_true += 1
    if num_total == 0:
        return 0
    return num_true/num_total
            
def get_num_categories(categories):
    if not categories:
        return 0
    categories = categories.split(',')
    return len(categories)

def get_num_checkins(checkin_data):
    return sum(checkin_data.values())

def get_yelping_since(yelping_since):
    date_obj = datetime.strptime(yelping_since, '%Y-%m-%d')
    utc_date = pytz.utc.localize(date_obj)
    return int(utc_date.timestamp())

def get_num_friends(friends):
    if friends == 'None':
        return 0
    friends = friends.split(',')
    return len(friends)

def get_num_elites(elite):
    if elite == 'None':
        return 0
    elite = elite.split(',')
    return len(elite)

# using the feature to index hashmap, create a vector of features for each business
# all categories are 1 hot encoded. 0 if doesnt exist, 1 if it does
# for each day in hours, convert the time slot into number of hours
# For boolean values, set it to 1 for True and 0 for False
# For categorical data, convert it to a deterministic hash value of 32 bit integer and use that

def decimal_hash32(s):
    ctx = decimal.getcontext()
    ctx.prec = 32
    h = decimal.Decimal(0)
    for c in s:
        h = (h * decimal.Decimal(131)) + decimal.Decimal(ord(c))
    return int(h % (2**32))

def get_feat_vector(data_row):
    
    bus_id = data_row['business_id']
    feature_vector = [None] * len(ALL_FEATURES)
    
    for k,v in data_row.items():
        
        # ignore useless features
        if k in {'business_id', 'name', 'neighborhood', 'address', 'city', 'postal_code'}:
            continue
        if type(v) != dict:
            # if it is categories, give it a value 1
            if k == 'categories':
                if v is not None:
                    categories = v.split(',')
                    for category in categories:
                        feature_name = category.strip()
                        feature_idx = ALL_FEATURES[feature_name]
                        feature_vector[feature_idx] = 1
            else:
                feature_name = k.strip()
                if feature_name not in {'hours', 'attributes'}:
                    feature_idx = ALL_FEATURES[feature_name]
                    
                    
                    
                    if v == True or v == 'True':
                        v = 1
                    elif v == False or v == 'False':
                        v = 0
                    elif type(v) == str:
                        v = decimal_hash32(v)
                    feature_vector[feature_idx] = v

        else:
            # if it is a dict then expand
            for k2, v2 in v.items():
                if v2[0] == '{' and v2[-1] == '}':
                    # convert to dict
                    v2 = ast.literal_eval(v2)

                    for k3, v3 in v2.items():
                        # use delimitter $ to rename feature
                        feature_name = k + '$' + k2 + '$' + k3
                        feature_name = feature_name.strip()
                        feature_idx = ALL_FEATURES[feature_name]
                        
                        if v3 == True or v3 == 'True':
                            v3 = 1
                        elif v3 == False or v3 == 'False':
                            v3 = 0
                        elif type(v3) == str:
                            v3 = decimal_hash32(v3)
                        feature_vector[feature_idx] = v3

                else:
                    # use delimitter $ to rename feature
                    feature_name = k + '$' + k2
                    feature_name = feature_name.strip()
                    feature_idx = ALL_FEATURES[feature_name]
                    
                    # change timeslot to hours for 'hours'
                    if k == 'hours':
                        feature_vector[feature_idx] = convert_timings_to_hours(v2)
                    else:
                        
                        if v2 == True or v2 == 'True':
                            v2 = 1
                        elif v2 == False or v2 == 'False':
                            v2 = 0
                        elif type(v2) == str:
                            v2 = decimal_hash32(v2)
                        
                        feature_vector[feature_idx] = v2
        
    return (bus_id, feature_vector)
#---------------------------------------------

# For each business, create its feature vector. (business_id, [feat1, feat2, .. feat1399])
business_RDD = sc.textFile(BUSINESS_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: get_feat_vector(x))

# Get the total number of check ins for a business
checkIn_RDD = sc.textFile(CHECKIN_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], get_num_checkins(x['time']))).map(lambda x: (x[0], [x[1]]))

# Get the total number of photos for a business
photo_RDD = sc.textFile(PHOTO_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))

# Get the total number of tips given by a user and the total number of tips for each business
tip_RDD = sc.textFile(TIP_FILE_PATH).map(lambda x: json.loads(x))

tips_business_RDD = tip_RDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))
tips_user_RDD = tip_RDD.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))

# Get the features for each user
user_RDD = sc.textFile(USER_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],
                                                                               [
                                                                                   int(x['review_count']),
                                                                                   get_yelping_since(x['yelping_since']),
                                                                                   get_num_friends(x['friends']),
                                                                                   int(x['useful']),
                                                                                   int(x['funny']),
                                                                                   int(x['cool']),
                                                                                   int(x['fans']),
                                                                                   get_num_elites(x['elite']),
                                                                                   float(x['average_stars']),
                                                                                   int(x['compliment_hot']),
                                                                                   int(x['compliment_more']),
                                                                                   int(x['compliment_profile']),
                                                                                   int(x['compliment_cute']),
                                                                                   int(x['compliment_list']),
                                                                                   int(x['compliment_note']),
                                                                                   int(x['compliment_plain']),
                                                                                   int(x['compliment_cool']),
                                                                                   int(x['compliment_funny']),
                                                                                   int(x['compliment_writer']),
                                                                                   int(x['compliment_photos'])
                                                                               ]))

def combine_lists(data_row):
    # fix nonetype error
    if data_row[1][1] == None:
        return[data_row[0], data_row[1][0] + [None]]
    if type(data_row[1][0]) == str:
        return [data_row[0], [data_row[1][0]] + data_row[1][1]]
    return [data_row[0], data_row[1][0] + data_row[1][1]]

# Combine the following RDDs to create a vector for each business with business id as key and list of features as value
# business_RDD + checkIn_RDD + photo_RDD + tips_business_RDD
# make sure to fix NoneType error when combining lists since some values are None

business_features_RDD = business_RDD.leftOuterJoin(checkIn_RDD).map(lambda x: combine_lists(x)).leftOuterJoin(photo_RDD).map(lambda x: combine_lists(x)).leftOuterJoin(tips_business_RDD).map(lambda x: combine_lists(x))

# Combine the following RDDs to create a vector for each user with user id as key and list of features as value
# user_RDD + tips_user_RDD
# make sure to fix NoneType error when combining lists since some values are None

user_features_RDD = user_RDD.leftOuterJoin(tips_user_RDD).map(lambda x: combine_lists(x))

def switch_keys(data_row):
    bus_id = data_row[0]
    usr_id = data_row[1][0]
    features = data_row[1][1:]
    
    return (usr_id, [bus_id] + features)

def join_all(data_row):
    usr_id = data_row[0]
    bus_id = data_row[1][0][0]
    bus_features = data_row[1][0][1:]
    usr_features = data_row[1][1]
    
    return ((usr_id, bus_id), bus_features + usr_features)

#----------- Testing Phase -----------
# Read in the testing dataset. Remove the header and convert a csv string into a list of 2 elements
# [user_id, business_id]
test_RDD = sc.textFile(TESTING_FILE_PATH)
headers_test = test_RDD.first()
test_RDD = test_RDD.filter(lambda x:x!=headers_test).map(lambda x:x.split(',')).map(lambda x:(x[0], x[1]))

# join the test_RDD and business_features_RDD
# we need to have the business_id as the key for this
test_RDD_tmp = test_RDD.map(lambda x: (x[1], x[0]))
test_join_business_features_RDD = test_RDD_tmp.leftOuterJoin(business_features_RDD).map(lambda x: combine_lists(x))

# now join this with the user_features_RDD. We need to have the user_id as key for this
test_join_business_features_RDD_tmp = test_join_business_features_RDD.map(lambda x: switch_keys(x))
test_join_business_features_user_features_RDD = test_join_business_features_RDD_tmp.leftOuterJoin(user_features_RDD)

# format the data as (user_id, business_id) [feature1, feature2, ...]
test_all_joined_RDD = test_join_business_features_user_features_RDD.map(lambda x: join_all(x))

def predictions_on_partition(part):
    
    test_partition_MAP = dict(part)
    
    # create the x testing list
    x_test_part = []
    test_labels_part = []
    for k in test_partition_MAP:
        x_test_part.append(test_partition_MAP[k])  
        test_labels_part.append(k)
    
    for xt in x_test_part:
        for i in range(len(xt)):
            if xt[i] is None:
                xt[i] = np.nan
    
    predictions = LOADED_MODEL.predict(data=x_test_part)
    predictions = [min(max(pred, 1.0), 5.0) for pred in predictions]
    
    output = []
    for i in range(len(test_labels_part)):
        output.append((test_labels_part[i][0], test_labels_part[i][1], predictions[i]))
    
    return output

predictions = test_all_joined_RDD.mapPartitions(lambda x: predictions_on_partition(x)).collect()

fhand = open(OUTPUT_FILE_PATH, 'w')
fhand.writelines('user_id, business_id, prediction\n')

for pred in predictions:
    fhand.writelines(pred[0] + ',' + pred[1] + ',' + str(pred[2]) + '\n')
fhand.close()

end_time = time.time()
print(f'Duration: {end_time-start_time}')
