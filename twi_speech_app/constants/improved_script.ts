import { RecordingSection, ScriptPrompt } from '@/types';

// SECTION A: Essential E-Commerce Voice Navigation Commands
// Focus on the most important actionable commands for navigating and making purchases
const scriptA: ScriptPrompt[] = [
  // App Navigation Commands
  { id: 'ScriptAU_1', type: 'scripted', text: 'Bue adetɔ app no', meaning: 'Open the shopping app' },
  { id: 'ScriptAU_2', type: 'scripted', text: 'Kɔ fie krataa so', meaning: 'Go to homepage' },
  { id: 'ScriptAU_3', type: 'scripted', text: 'Kɔ w\'anim', meaning: 'Go forward/Next' },
  { id: 'ScriptAU_4', type: 'scripted', text: 'San w\'akyi', meaning: 'Go back/Previous' },
  { id: 'ScriptAU_5', type: 'scripted', text: 'Fa kɔ kart no mu', meaning: 'Go to cart' },
  { id: 'ScriptAU_6', type: 'scripted', text: 'Kɔ soro', meaning: 'Scroll up' },
  { id: 'ScriptAU_7', type: 'scripted', text: 'Kɔ fam', meaning: 'Scroll down' },
  { id: 'ScriptAU_8', type: 'scripted', text: 'Kɔ me akaont no mu', meaning: 'Go to my account' },
  { id: 'ScriptAU_9', type: 'scripted', text: 'Hwɛ me kart no mu nneɛma', meaning: 'View items in my cart' },

  // Product Search Commands
  { id: 'ScriptAU_10', type: 'scripted', text: 'Hwehwɛ atadeɛ', meaning: 'Search for clothing' },
  { id: 'ScriptAU_11', type: 'scripted', text: 'Hwehwɛ mfonini hwɛdeɛ', meaning: 'Search for electronics' },
  { id: 'ScriptAU_12', type: 'scripted', text: 'Hwehwɛ nhoma', meaning: 'Search for books' },
  { id: 'ScriptAU_13', type: 'scripted', text: 'Hwehwɛ telefon', meaning: 'Search for phones' },
  { id: 'ScriptAU_14', type: 'scripted', text: 'Hwehwɛ aduane', meaning: 'Search for food items' },
  { id: 'ScriptAU_15', type: 'scripted', text: 'Fa di nsanasin mu', meaning: 'Apply filter' },

  // Product Interaction Commands
  { id: 'ScriptAU_16', type: 'scripted', text: 'Fa wei gu me kart no mu', meaning: 'Add this to my cart' },
  { id: 'ScriptAU_17', type: 'scripted', text: 'Yi wei firi kart no mu', meaning: 'Remove this from cart' },
  { id: 'ScriptAU_18', type: 'scripted', text: 'Twerɛ m\'adwen fa yei ho', meaning: 'Write a review for this' },
  { id: 'ScriptAU_19', type: 'scripted', text: 'Kenkan adwadeɛ no ho nsɛm ma me', meaning: 'Read the product details to me' },
  { id: 'ScriptAU_20', type: 'scripted', text: 'Hwehwɛ adwadeɛ serɛ yi', meaning: 'Find similar products' },

  // Purchase Commands
  { id: 'ScriptAU_21', type: 'scripted', text: 'Fa wie tɔneɛ', meaning: 'Proceed to checkout' },
  { id: 'ScriptAU_22', type: 'scripted', text: 'Tua ka seesei ara', meaning: 'Pay now' },
  { id: 'ScriptAU_23', type: 'scripted', text: 'Yi fa mobile money tua ka', meaning: 'Select mobile money payment' },
  { id: 'ScriptAU_24', type: 'scripted', text: 'Yi fa akonhoma so tua ka', meaning: 'Select card payment' },
  { id: 'ScriptAU_25', type: 'scripted', text: 'Hyɛ da a merebɛgye adwadeɛ no', meaning: 'Select a delivery date' },

  // Help and Support Commands
  { id: 'ScriptAU_26', type: 'scripted', text: 'Boa me wɔ adwadeɛ yi ho', meaning: 'Help me with this product' },
  { id: 'ScriptAU_27', type: 'scripted', text: 'Kyerɛkyerɛ mu ma me', meaning: 'Explain this to me' },
  { id: 'ScriptAU_28', type: 'scripted', text: 'Frɛ mmoa kuw no', meaning: 'Call customer support' },
  { id: 'ScriptAU_29', type: 'scripted', text: 'Kyerɛ me kwan ma mentɔ adwadeɛ', meaning: 'Guide me through purchasing a product' },
  { id: 'ScriptAU_30', type: 'scripted', text: 'Ɛhe na mehunu atɔdeɛ ahyehyɛdeɛ?', meaning: 'Where can I find order settings?' },

  // Error Handling Commands
  { id: 'ScriptAU_31', type: 'scripted', text: 'Adɛn nti na mentumi mfa mmfa kart no mu?', meaning: 'Why can\'t I add it to cart?' },
  { id: 'ScriptAU_32', type: 'scripted', text: 'Me werɛ afi me akwankyerɛ nsɛm', meaning: 'I have forgotten my password' },
  { id: 'ScriptAU_33', type: 'scripted', text: 'Sakra me kwan nkyerɛdeɛ', meaning: 'Change my address' },
  { id: 'ScriptAU_34', type: 'scripted', text: 'Yi saa mfomsoɔ yi', meaning: 'Fix this error' },
  { id: 'ScriptAU_35', type: 'scripted', text: 'Mente deɛ woreka no aseɛ', meaning: 'I don\'t understand what you are saying' },

  // Accessibility Commands
  { id: 'ScriptAU_36', type: 'scripted', text: 'Kenkan krataa yi mu nneɛma nyinaa', meaning: 'Read everything on this page' },
  { id: 'ScriptAU_37', type: 'scripted', text: 'Yɛ nsɛnkyerɛnnee no kɛse', meaning: 'Make the text larger' },
  { id: 'ScriptAU_38', type: 'scripted', text: 'Ma kasa no so', meaning: 'Increase volume' },
  { id: 'ScriptAU_39', type: 'scripted', text: 'Yɛ kasa no brɛoo', meaning: 'Speak slower' },
  { id: 'ScriptAU_40', type: 'scripted', text: 'Kyerɛ me mfonini mu nneɛma', meaning: 'Describe the items in the image' },

  // Confirmation Commands
  { id: 'ScriptAU_41', type: 'scripted', text: 'Aane, mepene so', meaning: 'Yes, I confirm' },
  { id: 'ScriptAU_42', type: 'scripted', text: 'Daabi, gyae', meaning: 'No, stop' },
  { id: 'ScriptAU_43', type: 'scripted', text: 'Siesie m\'atɔ ahyɛnsodeɛ no', meaning: 'Complete my purchase' },
  { id: 'ScriptAU_44', type: 'scripted', text: 'Pɛ saa adwadeɛ yi ma me', meaning: 'Select this product for me' },
  { id: 'ScriptAU_45', type: 'scripted', text: 'Medi kan ahwɛ nneɛma yi mu bio', meaning: 'I want to review these items again' },
];

// SECTION B: Numbers and Quantities for E-commerce
// Focused specifically on quantities, prices, and numerical expressions needed for shopping
const scriptB: ScriptPrompt[] = [
  // Basic Numbers for Product Quantities
  { id: 'ScriptBU_1', type: 'scripted', text: 'Baako', meaning: 'One' },
  { id: 'ScriptBU_2', type: 'scripted', text: 'Mmienu', meaning: 'Two' },
  { id: 'ScriptBU_3', type: 'scripted', text: 'Mmiɛnsa', meaning: 'Three' },
  { id: 'ScriptBU_4', type: 'scripted', text: 'Ɛnan', meaning: 'Four' },
  { id: 'ScriptBU_5', type: 'scripted', text: 'Enum', meaning: 'Five' },

  // Product Quantity Actions
  { id: 'ScriptBU_6', type: 'scripted', text: 'Mepɛ baako pɛ', meaning: 'I want only one' },
  { id: 'ScriptBU_7', type: 'scripted', text: 'Fa mmienu gu kart no mu', meaning: 'Add two to the cart' },
  { id: 'ScriptBU_8', type: 'scripted', text: 'Yi baako firi mu', meaning: 'Remove one from it' },
  { id: 'ScriptBU_9', type: 'scripted', text: 'Mepɛ adwadeɛ yi mmiɛnsa', meaning: 'I want three of this item' },
  { id: 'ScriptBU_10', type: 'scripted', text: 'Yi ne mmienu ka ho', meaning: 'Add two more of it' },

  // Price Commands and Queries
  { id: 'ScriptBU_11', type: 'scripted', text: 'Adwadeɛ yi boɔ yɛ sɛn?', meaning: 'How much is this product?' },
  { id: 'ScriptBU_12', type: 'scripted', text: 'Kyerɛ me nneɛma a ɛnsene sidi ɔha', meaning: 'Show me items under 100 cedis' },
  { id: 'ScriptBU_13', type: 'scripted', text: 'Hwehwɛ nneɛma a ɛdi sidi aduonu ne sidi aduonum ntam', meaning: 'Search for items between 20 and 50 cedis' },
  { id: 'ScriptBU_14', type: 'scripted', text: 'Te boɔ no so ma me', meaning: 'Give me a discount' },
  { id: 'ScriptBU_15', type: 'scripted', text: 'Mɛtua sidi ɔha ne aduasa enum', meaning: 'I will pay one hundred and thirty-five cedis' },

  // Delivery Time Commands
  { id: 'ScriptBU_16', type: 'scripted', text: 'Mepɛ adwadeɛ no nnɔnhwere mmienu mu', meaning: 'I want the product within two hours' },
  { id: 'ScriptBU_17', type: 'scripted', text: 'Mepɛ express delivery', meaning: 'I want express delivery' },
  { id: 'ScriptBU_18', type: 'scripted', text: 'Adwadeɛ no bɛba daben?', meaning: 'When will the product arrive?' },
  { id: 'ScriptBU_19', type: 'scripted', text: 'Mepɛ adwadeɛ no ɛnnɛ anɔpa', meaning: 'I want the product this morning' },
  { id: 'ScriptBU_20', type: 'scripted', text: 'Fa akɔ same-day delivery mu', meaning: 'Select same-day delivery' },

  // Specific Quantities for Different Products
  { id: 'ScriptBU_21', type: 'scripted', text: 'Mepɛ atadeɛ enum', meaning: 'I want five clothes' },
  { id: 'ScriptBU_22', type: 'scripted', text: 'Fa telefon mmienu ka ho', meaning: 'Add two phones' },
  { id: 'ScriptBU_23', type: 'scripted', text: 'Mepɛ aduane mpakuo nson', meaning: 'I want seven food packages' },
  { id: 'ScriptBU_24', type: 'scripted', text: 'Fa laptop baako gu mu', meaning: 'Add one laptop to it' },
  { id: 'ScriptBU_25', type: 'scripted', text: 'Mepɛ nhoma mmienu', meaning: 'I want two books' },

  // Payment Amounts and Options
  { id: 'ScriptBU_26', type: 'scripted', text: 'Ɛyɛ Ghana sidi ɔha', meaning: 'It is one hundred Ghana Cedis' },
  { id: 'ScriptBU_27', type: 'scripted', text: 'Ɛyɛ sidi apem ne ahannum', meaning: 'It is one thousand five hundred cedis' },
  { id: 'ScriptBU_28', type: 'scripted', text: 'Tua nkyekyɛmu aduonu ɔha mu seesei', meaning: 'Pay 20 percent now' },
  { id: 'ScriptBU_29', type: 'scripted', text: 'Mepɛ sɛ metua wɔ asetena mu nkyekyɛmu mmiɛnsa', meaning: 'I want to pay in three installments' },
  { id: 'ScriptBU_30', type: 'scripted', text: 'Sika a wɔbɛgye ama adwadeɛ no nya krakra yɛ sidi aduonu', meaning: 'The delivery fee is twenty cedis' },

  // Order Status with Numbers
  { id: 'ScriptBU_31', type: 'scripted', text: 'M\'atɔdeɛ nɔma edunsia yi kɔɔ sɛn?', meaning: 'What is the status of my order number sixteen?' },
  { id: 'ScriptBU_32', type: 'scripted', text: 'Adwadeɛ no wɔ amanneɛbɔ nkyekyɛmu mmienu mu', meaning: 'The product is in stage two of processing' },
  { id: 'ScriptBU_33', type: 'scripted', text: 'Wo wɔ adwadeɛ enum wɔ wo kart no mu', meaning: 'You have five products in your cart' },
  { id: 'ScriptBU_34', type: 'scripted', text: 'Mepɛ Express Delivery a ɛyɛ sidi aduonum', meaning: 'I want the Express Delivery that costs fifty cedis' },
  { id: 'ScriptBU_35', type: 'scripted', text: 'Mennsɛ sɛ metua sidi ɔha ne aduonu', meaning: 'I don\'t agree to pay one hundred and twenty cedis' },

  // Time-Based Shopping Commands
  { id: 'ScriptBU_36', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Ɛdwoada', meaning: 'You will get your product on Monday' },
  { id: 'ScriptBU_37', type: 'scripted', text: 'Wo adwadeɛ no bɛba dɔnhwere mmiɛnsa mu', meaning: 'Your product will arrive in three hours' },
  { id: 'ScriptBU_38', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi bosome a atwam', meaning: 'I want my purchase history from last month' },
  { id: 'ScriptBU_39', type: 'scripted', text: 'Hwehwɛ adwadeɛ a atwa mu wɔ nnaawɔtwe mmienu a atwam mu', meaning: 'Find products purchased in the past two weeks' },
  { id: 'ScriptBU_40', type: 'scripted', text: 'Twe saa adwadeɛ no kɔsi nnafua nkron mu', meaning: 'Delay that product for nine days' },

  // Rating and Reviews with Numbers
  { id: 'ScriptBU_41', type: 'scripted', text: 'Ma adwadeɛ no nsoromma enum', meaning: 'Give the product five stars' },
  { id: 'ScriptBU_42', type: 'scripted', text: 'Hwehwɛ adwadeɛ a wɔama no nsoromma ɛnan', meaning: 'Search for products with four-star ratings' },
  { id: 'ScriptBU_43', type: 'scripted', text: 'Kyerɛ me awerɛkyekyerɛ a wɔatwerɛ ho nneɛma edu', meaning: 'Show me the ten most recent reviews' },
  { id: 'ScriptBU_44', type: 'scripted', text: 'Adwadeɛ yi nya amanneɛbɔ ɔha mu nkyekyɛmu aduɔwɔtwe', meaning: 'This product has an eighty percent approval rating' },
  { id: 'ScriptBU_45', type: 'scripted', text: 'Hwehwɛ adwadeɛ a wɔatɔ no mpɛn apem', meaning: 'Search for products that have been purchased a thousand times' },
];

// SECTION C: E-commerce Practical Shopping Scenarios
// Focused on realistic shopping interactions that apply to various products
const scriptC: ScriptPrompt[] = [
  // Product Category Navigation
  { id: 'ScriptCU_1', type: 'scripted', text: 'Fa me kɔ atadeɛ kart so', meaning: 'Take me to the clothing section' },
  { id: 'ScriptCU_2', type: 'scripted', text: 'Fa me kɔ mfonini hwɛdeɛ kart so', meaning: 'Take me to the electronics section' },
  { id: 'ScriptCU_3', type: 'scripted', text: 'Fa me kɔ aduane kart so', meaning: 'Take me to the grocery section' },
  { id: 'ScriptCU_4', type: 'scripted', text: 'Fa me kɔ fie nneɛma kart so', meaning: 'Take me to household items section' },
  { id: 'ScriptCU_5', type: 'scripted', text: 'Fa me kɔ mmofra kart so', meaning: 'Take me to kids section' },

  // Product Specification Queries
  { id: 'ScriptCU_6', type: 'scripted', text: 'Kyerɛ me telefon yi ahyɛnsodeɛ', meaning: 'Show me this phone\'s specifications' },
  { id: 'ScriptCU_7', type: 'scripted', text: 'Wei yɛ den dodo', meaning: 'This is too expensive' },
  { id: 'ScriptCU_8', type: 'scripted', text: 'Wowɔ yei wɔ kɔlɔ foforɔ mu?', meaning: 'Do you have this in other colors?' },
  { id: 'ScriptCU_9', type: 'scripted', text: 'Mepɛ kɛtɛ a ɛso kakra', meaning: 'I want a slightly larger size' },
  { id: 'ScriptCU_10', type: 'scripted', text: 'Adwadeɛ yi yɛ adwuma sɛn?', meaning: 'How does this product work?' },

  // Product Comparison Commands
  { id: 'ScriptCU_11', type: 'scripted', text: 'Fa yei ne wei to nkorɛnkorɛ', meaning: 'Compare these two items' },
  { id: 'ScriptCU_12', type: 'scripted', text: 'Kyerɛ me nsonsonoeɛ a ɛda yei ne yei ntam', meaning: 'Show me the difference between this and this' },
  { id: 'ScriptCU_13', type: 'scripted', text: 'Deɛn na ɛyɛ papa wɔ yei mu?', meaning: 'What\'s better about this one?' },
  { id: 'ScriptCU_14', type: 'scripted', text: 'Deɛ ɛwɔ he na adefoɔ pɛ no paa?', meaning: 'Which one is most popular?' },
  { id: 'ScriptCU_15', type: 'scripted', text: 'Hwehwɛ adwadeɛ serɛ yi ne boɔ a ɛsɛ', meaning: 'Find similar products at a lower price' },

  // Shopping Cart Management
  { id: 'ScriptCU_16', type: 'scripted', text: 'Fa wei ka me kart ho', meaning: 'Add this to my cart' },
  { id: 'ScriptCU_17', type: 'scripted', text: 'Yi yei firi me kart no mu', meaning: 'Remove this from my cart' },
  { id: 'ScriptCU_18', type: 'scripted', text: 'Sesa aduane no dodow wɔ me kart no mu', meaning: 'Change the quantity of food in my cart' },
  { id: 'ScriptCU_19', type: 'scripted', text: 'Yi me kart no mu nneɛma nyinaa', meaning: 'Clear all items in my cart' },
  { id: 'ScriptCU_20', type: 'scripted', text: 'Fa me kɔ awiee tɔ mu afiri kart yi mu', meaning: 'Proceed to checkout from this cart' },

  // Delivery Options
  { id: 'ScriptCU_21', type: 'scripted', text: 'Kyerɛ me akwanya ahorow a mewɔ wɔ akra ho', meaning: 'Show me the available delivery options' },
  { id: 'ScriptCU_22', type: 'scripted', text: 'Mepɛ sɛ wode brɛ me ntɛm', meaning: 'I want it delivered quickly' },
  { id: 'ScriptCU_23', type: 'scripted', text: 'Mɛgye adwadeɛ no wɔ store no mu', meaning: 'I\'ll pick up the product at the store' },
  { id: 'ScriptCU_24', type: 'scripted', text: 'Wɔde bɛkra me ɛhe?', meaning: 'Where will it be delivered?' },
  { id: 'ScriptCU_25', type: 'scripted', text: 'Sesa delivery address no', meaning: 'Change the delivery address' },

  // Payment Process
  { id: 'ScriptCU_26', type: 'scripted', text: 'Mɛtumi de mobile money atua ka?', meaning: 'Can I pay using mobile money?' },
  { id: 'ScriptCU_27', type: 'scripted', text: 'Fa me kɔ tua ka kwan so', meaning: 'Take me to payment methods' },
  { id: 'ScriptCU_28', type: 'scripted', text: 'Fa tua ka no sie', meaning: 'Save this payment method' },
  { id: 'ScriptCU_29', type: 'scripted', text: 'Tua ka seesei', meaning: 'Pay now' },
  { id: 'ScriptCU_30', type: 'scripted', text: 'Merentua ka seesei', meaning: 'I\'m not paying now' },

  // Order Tracking
  { id: 'ScriptCU_31', type: 'scripted', text: 'M\'adwadeɛ no wɔ he seesei?', meaning: 'Where is my product now?' },
  { id: 'ScriptCU_32', type: 'scripted', text: 'Kwan bɛn so na metumi ahwɛ m\'adetɔ ahyɛnsodeɛ?', meaning: 'How can I track my order?' },
  { id: 'ScriptCU_33', type: 'scripted', text: 'M\'adwadeɛ no adu?', meaning: 'Has my order arrived?' },
  { id: 'ScriptCU_34', type: 'scripted', text: 'M\'adwadeɛ no abɛn?', meaning: 'Is my order close to arriving?' },
  { id: 'ScriptCU_35', type: 'scripted', text: 'Adɛn nti na me nnsaa nnya m\'adwadeɛ no?', meaning: 'Why haven\'t I received my product yet?' },

  // Return and Exchange
  { id: 'ScriptCU_36', type: 'scripted', text: 'Mepɛ sɛ mesan de adwadeɛ yi ma mo', meaning: 'I want to return this product' },
  { id: 'ScriptCU_37', type: 'scripted', text: 'Adwadeɛ a wɔde brɛɛ me no ayɛ basabasa', meaning: 'The delivered product is damaged' },
  { id: 'ScriptCU_38', type: 'scripted', text: 'Mepɛ sɛ mesesa adwadeɛ yi', meaning: 'I want to exchange this product' },
  { id: 'ScriptCU_39', type: 'scripted', text: 'Kyerɛ me sɛnea meyi adwadeɛ yi akɔ', meaning: 'Show me how to return this product' },
  { id: 'ScriptCU_40', type: 'scripted', text: 'Wobɛtumi ayi sika firi me akaont no mu ama me?', meaning: 'Can you process a refund for me?' },
];

// SECTION D: Complex Shopping Dialogues and Scenarios
// Focusing on more complex interactions and problems that may occur in e-commerce
const scriptD: ScriptPrompt[] = [
  // Account Management and Security
  { id: 'ScriptDU_1', type: 'scripted', text: 'Me werɛ afi me akaont ho kodeɛ', meaning: 'I\'ve forgotten my account password' },
  { id: 'ScriptDU_2', type: 'scripted', text: 'Boa me ma mensesa me afidie ho nsɛm', meaning: 'Help me change my device information' },
  { id: 'ScriptDU_3', type: 'scripted', text: 'Sakra me phone number wɔ m\'akaont no mu', meaning: 'Change my phone number in my account' },
  { id: 'ScriptDU_5', type: 'scripted', text: 'Merepɛ sɛ menya obi a ɔbɛtumi ahwɛ m\'akaont', meaning: 'I want to add someone who can manage my account' },

  // Complex Payment Scenarios
  { id: 'ScriptDU_6', type: 'scripted', text: 'Sɛ metua sika seesei a, mɛnya atɔ mmara mu aduasa ɔha mu', meaning: 'If I pay now, I\'ll receive a 30% discount' },
  { id: 'ScriptDU_7', type: 'scripted', text: 'Sɛ wo akonhoma no nni sika a, yɛntumi ntɔ adwadeɛ no', meaning: 'If your card has no money, we cannot purchase the product' },
  { id: 'ScriptDU_8', type: 'scripted', text: 'Mepɛ sɛ mede m\'akyɛdeɛ krataa tua sidi ahannum ne aduasa', meaning: 'I want to use my gift card to pay five hundred and thirty cedis' },
  { id: 'ScriptDU_9', type: 'scripted', text: 'M\'akonhoma no wɔagye ho ban wɔ amanɔne adetɔ ho', meaning: 'My card is blocked for international purchases' },
  { id: 'ScriptDU_10', type: 'scripted', text: 'Mɛpɛ sɛ metua sika no nkyekyɛmu mmienu so - mobile money ne akonhoma', meaning: 'I\'d like to split the payment - mobile money and card' },

  // Order Problems and Resolutions
  { id: 'ScriptDU_11', type: 'scripted', text: 'Adwadeɛ no nyɛ adwuma firi nnora', meaning: 'The product hasn\'t been working since yesterday' },
  { id: 'ScriptDU_12', type: 'scripted', text: 'Wɔde adwadeɛ a ɛnyɛ me deɛ no ama me', meaning: 'I received a product that\'s not what I ordered' },
  { id: 'ScriptDU_13', type: 'scripted', text: 'Me hia teknikal mmoa ntɛm ara', meaning: 'I need technical support immediately' },
  { id: 'ScriptDU_14', type: 'scripted', text: 'Adwadeɛ a wɔde maeɛ no annu mu', meaning: 'The product delivered is incomplete' },
  { id: 'ScriptDU_15', type: 'scripted', text: 'Me werɛ aho sɛ adwadeɛ a metɔeɛ no anyɛ adwuma', meaning: 'I am very sad that the product I bought is not working' },

  // Multi-Item Shopping Coordination
  { id: 'ScriptDU_16', type: 'scripted', text: 'Fa atadeɛ yi ne mpaboa a ɛfata to ho', meaning: 'Match this clothing with suitable shoes' },
  { id: 'ScriptDU_17', type: 'scripted', text: 'Mepɛ nneɛma a mede bɛhyehyɛ me dan foforɔ no', meaning: 'I want items to furnish my new room' },
  { id: 'ScriptDU_18', type: 'scripted', text: 'Hwehwɛ aduane mmoawa a mɛtumi de ayɛ apiti anɔpa', meaning: 'Find ingredients I can use to make breakfast' },
  { id: 'ScriptDU_19', type: 'scripted', text: 'Fa nneɛma a wɔde di aponto gu me kart no mu', meaning: 'Add party supplies to my cart' },
  { id: 'ScriptDU_20', type: 'scripted', text: 'Kyerɛ me nneɛma a wɔtaa tɔ no abom', meaning: 'Show me items commonly bought together' },

  // Advanced Product Search
  { id: 'ScriptDU_21', type: 'scripted', text: 'Hwehwɛ telefon a ɛwɔ camera papa nanso ɛnyɛ den', meaning: 'Search for phones with good cameras but not expensive' },
  { id: 'ScriptDU_22', type: 'scripted', text: 'Kyerɛ me adwadeɛ a nnipa a wɔtaa tɔ telefon taa hwɛ', meaning: 'Show me products that phone buyers frequently view' },
  { id: 'ScriptDU_23', type: 'scripted', text: 'Hwehwɛ atadeɛ a wɔapam ho nsuo', meaning: 'Search for water-resistant clothing' },
  { id: 'ScriptDU_24', type: 'scripted', text: 'Hwehwɛ aduane a wɔnnyaa ahahan wɔ mu', meaning: 'Search for food without preservatives' },
  { id: 'ScriptDU_25', type: 'scripted', text: 'Kyerɛ me nneɛma a wɔatɔn pii nnansa yi', meaning: 'Show me the best-selling items recently' },

  // Voice Shopping with Accessibility Needs
  { id: 'ScriptDU_26', type: 'scripted', text: 'Me pɛ adetɔ dwumadi a ɛboa wɔn a wɔnhunu adeɛ', meaning: 'I prefer shopping platforms that help those with visual impairments' },
  { id: 'ScriptDU_27', type: 'scripted', text: 'Menntumi nnkenkan nneɛma no yie, enti kenkan ma me', meaning: 'I can\'t read the items well, so read for me' },
  { id: 'ScriptDU_28', type: 'scripted', text: 'Me nsateaa ntumi nnyɛ adwuma yie, mede me nne nko ara bɛtɔ adeɛ', meaning: 'My fingers don\'t work well, I\'ll shop using only my voice' },
  { id: 'ScriptDU_29', type: 'scripted', text: 'Kyerɛ me adwadeɛ a ɛbɛboa me a mewɔ aniwa haw', meaning: 'Show me products that will help me with my vision problem' },
  { id: 'ScriptDU_30', type: 'scripted', text: 'Kenkan awerɛkyekyerɛ no ma me brɛoo na mente aseɛ', meaning: 'Read the reviews slowly so I can understand' },

  // Advanced Platform Navigation
  { id: 'ScriptDU_31', type: 'scripted', text: 'Kyerɛ me akwan a metumi agyina so ahwehwɛ adwadeɛ wɔ dwumadi yi so', meaning: 'Show me different ways to search for products on this platform' },
  { id: 'ScriptDU_32', type: 'scripted', text: 'Sɔ kwan nhyehyɛeɛ a me fa so di dwuma mmerɛ dodo', meaning: 'Change to a navigation setting that I use most often' },
  { id: 'ScriptDU_33', type: 'scripted', text: 'Fa me kɔ adetɔ akwan a mefaa so dada no so', meaning: 'Take me to my previous shopping paths' },
  { id: 'ScriptDU_34', type: 'scripted', text: 'Kyerɛ me sɛnea mesi mira adeɛ a mepɛ', meaning: 'Show me how to bookmark items I\'m interested in' },
  { id: 'ScriptDU_35', type: 'scripted', text: 'Boa me ma me nsusu nneɛma ahorow ahyehyɛ no', meaning: 'Help me understand the category organization' },
];

// SECTION E: Spontaneous E-commerce Speech Scenarios
// Open-ended scenarios that will encourage natural speech about e-commerce situations
const spontaneousPrompts: ScriptPrompt[] = [
  // General Shopping Experience
  { id: 'SpontaneousU_1', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wode wo nne di intanɛte adetɔ ho dwuma - nneɛma a wopɛ ɛne deɛ ɛyɛ mmerɛw ma wo. (Explain how you use voice for online shopping - what you like and what comes easily to you.)' },

  // Technical Challenges
  { id: 'SpontaneousU_2', type: 'spontaneous', text: 'Ka sɛnea ɛyɛ den ma wo sɛ wode wo nne bɛhyehyɛ nneɛma wɔ online store mu, ɛne sɛnea wobɛpɛ sɛ wɔsiesie no. (Talk about the challenges you face using voice to navigate online stores, and how you would like them improved.)' },

  // Voice Assistant Experience
  { id: 'SpontaneousU_3', type: 'spontaneous', text: 'Kyerɛkyerɛ wo nkasa mmoa dwumadi a ɛdi mu pa ara a wobɛpɛ ama e-commerce - sua ne ahoɔden a wopɛ sɛ ɛbɛwɔ. (Describe your ideal voice assistant for e-commerce - features and capabilities you wish it had.)' },

  // Shopping on Behalf of Others
  { id: 'SpontaneousU_4', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛtumi de wo nne ato adeɛ ama wo busuani a ɔwɔ sinnaeɛ bi a ɔntumi ntɔ adeɛ ankasa. (Explain how you would use voice to shop for a relative who has a condition preventing them from shopping themselves.)' },

  // Complex Shopping Decision
  { id: 'SpontaneousU_5', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wode wo nne bɛhwehwɛ, akɔ mu ahwɛ, na ɛtwe atwa adwadeɛ a ne boɔ yɛ den te sɛ telefon anaa laptop. (Describe how you would use voice to research, compare, and finally purchase an expensive item like a phone or laptop.)' },

  // Emergency Shopping
  { id: 'SpontaneousU_6', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛfa so de wo nne atɔ adeɛ wɔ kasiɛ mu te sɛ sɛ ɛsɛ sɛ wotɔ aduro mmerɛ a yareɛ bi akɔ so. (Explain how you would use voice shopping in an emergency like needing to purchase medicine during an illness.)' },

  // Payment Security Concerns
  { id: 'SpontaneousU_7', type: 'spontaneous', text: 'Kyerɛkyerɛ wo nsuro fa sika tua bammɔ ho berɛ a wode wo nne retɔ adeɛ wɔ intanɛte so, ɛne sɛnea wobɛpɛ sɛ wɔbɔ ho ban. (Describe your concerns about payment security when using voice for online shopping, and how you would like it secured.)' },

  // Shopping for Special Occasions
  { id: 'SpontaneousU_8', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛfa so de nkasa ahyehyɛdeɛ atɔ akyɛdeɛ ama obi wɔ ne awoda, a wopɛ sɛ ɛyɛ nwanwa. (Explain how you would use a voice system to buy a surprise gift for someone\'s birthday.)' },

  // Troubleshooting Failed Orders
  { id: 'SpontaneousU_9', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛfa wo nne agyina so asiesie ɔhaw bi wɔ adetɔ a wannkɔ yie ho - te sɛ adwadeɛ a wɔamfa amma, anaa woade adwadeɛ foforɔ abrɛ wo. (Explain how you would use voice to resolve an issue with a failed order - like a missing delivery or wrong product.)' },

  // Shopping and Budget Management
  { id: 'SpontaneousU_10', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛfa so de nkasa ahyehyɛdeɛ ahwɛ wo sika a wode bɛtɔ adeɛ so berɛ a woretɔ nneɛma pii wɔ intanɛte so. (Describe how you would use voice commands to manage your budget while shopping for multiple items online.)' },

  // Comparing Multiple Products
  { id: 'SpontaneousU_11', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wobɛpɛ sɛ wode wo nne fa so de to mfonini hwɛdeɛ ahorow nkorɛnkorɛ - te sɛ mfonini hwɛdeɛ ahorow mmiɛnsa anaa ɛnnan. (Describe how you would like to use voice to compare multiple electronic devices - such as three or four different models.)' },

  // Browsing without Specific Goals
  { id: 'SpontaneousU_12', type: 'spontaneous', text: 'Kyerɛkyerɛ sɛnea wode wo nne bɛdi dwuma ahwehwɛ nneɛma wɔ intanɛte so berɛ a wunnhunuu pɔtee deɛ worehwehwɛ. (Explain how you would use voice commands to browse online when you don\'t have a specific item in mind.)' },
];

// --- Define Sections ---
export const RECORDING_SECTIONS: RecordingSection[] = [
  {
    id: 'ScriptAU',
    title: 'Essential E-Commerce Voice Commands',
    description: 'Read these common Twi voice commands for navigating and making purchases on e-commerce platforms.',
    prompts: scriptA,
  },
  {
    id: 'ScriptBU',
    title: 'Numbers and Quantities for Shopping',
    description: 'Read these phrases about numbers, quantities, prices, and timing related to online shopping.',
    prompts: scriptB,
  },
  {
    id: 'ScriptCU',
    title: 'Practical Online Shopping Scenarios',
    description: 'Read these phrases covering common actions when shopping online for various products.',
    prompts: scriptC,
  },
  {
    id: 'ScriptDU',
    title: 'Complex E-Commerce Interactions',
    description: 'Read these more complex phrases for handling special situations during online shopping.',
    prompts: scriptD,
  },
  {
    id: 'SpontaneousU',
    title: 'Voice Shopping Experiences',
    description: 'For the next prompts, read the topic about voice shopping, then speak freely and naturally about it in Twi for 30-60 seconds. DO NOT read the instructions aloud.',
    prompts: spontaneousPrompts,
  }
];

export const getTotalPrompts = () => {
  let total = 0;
  RECORDING_SECTIONS.forEach(section => {
    total += section.prompts.length;
  });
  return total;
};

export const EXPECTED_TOTAL_RECORDINGS = getTotalPrompts();
export const SPONTANEOUS_PROMPTS_COUNT = spontaneousPrompts.length;

export const getGlobalPromptIndex = (sectionIndex: number, promptInSectionIndex: number): number => {
  let globalIndex = 0;
  for (let i = 0; i < sectionIndex; i++) {
    globalIndex += RECORDING_SECTIONS[i].prompts.length;
  }
  globalIndex += promptInSectionIndex;
  return globalIndex;
};
