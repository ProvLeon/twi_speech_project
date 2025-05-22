import { RecordingSection, ScriptPrompt } from '@/types';

const scriptA: ScriptPrompt[] = [
  { id: 'ScriptA_1', type: 'scripted', text: 'Maakye', meaning: 'Good morning' },
  { id: 'ScriptA_2', type: 'scripted', text: 'Maaha', meaning: 'Good afternoon' },
  { id: 'ScriptA_3', type: 'scripted', text: 'Maadwo', meaning: 'Good evening/night' },
  { id: 'ScriptA_4', type: 'scripted', text: 'Wo ho te sɛn?', meaning: 'How are you?' },
  { id: 'ScriptA_5', type: 'scripted', text: 'Me ho yɛ.', meaning: 'I am fine.' },
  { id: 'ScriptA_6', type: 'scripted', text: 'Bɔkɔɔ.', meaning: 'Fine / Cool / Okay.' },
  { id: 'ScriptA_7', type: 'scripted', text: 'Medaase', meaning: 'Thank you.' },
  { id: 'ScriptA_8', type: 'scripted', text: 'Medaase paa.', meaning: 'Thank you very much.' },
  { id: 'ScriptA_9', type: 'scripted', text: 'Mepɛ sɛ metɔ biribi.', meaning: 'I want to buy something.' },
  { id: 'ScriptA_10', type: 'scripted', text: 'Metumi akɔtwe sika?', meaning: 'Can I withdraw money?' },
  { id: 'ScriptA_11', type: 'scripted', text: 'Mesrɛ wo, bue me akantabuo.', meaning: 'Please, open an account for me.' },
  { id: 'ScriptA_12', type: 'scripted', text: 'Hwehwɛ atɔdeɛ.', meaning: 'Search for products.' },
  { id: 'ScriptA_13', type: 'scripted', text: 'Fa kɔ nkrataa mu.', meaning: 'Go to cart.' },
  { id: 'ScriptA_14', type: 'scripted', text: 'Kɔ w'anim.', meaning: 'Go forward/ Next.' },
  { id: 'ScriptA_15', type: 'scripted', text: 'San w'akyi.', meaning: 'Go back/ Previous.' },
  { id: 'ScriptA_16', type: 'scripted', text: 'Fa gu me nkrataa mu.', meaning: 'Add to my cart.' },
  { id: 'ScriptA_17', type: 'scripted', text: 'Yi firi nkrataa mu.', meaning: 'Remove from cart.' },
  { id: 'ScriptA_18', type: 'scripted', text: 'Me pɛ sɛ mewie tɔneɛ.', meaning: 'I want to checkout.' },
  { id: 'ScriptA_19', type: 'scripted', text: 'Sɛn na metumi atua ka?', meaning: 'How can I pay?' },
  { id: 'ScriptA_20', type: 'scripted', text: 'Wɔde bɛkra me ɛhe?', meaning: 'Where will it be delivered?' },
  { id: 'ScriptA_21', type: 'scripted', text: 'Ɛyɛ.', meaning: 'It is good / Okay.' },
  { id: 'ScriptA_22', type: 'scripted', text: 'Ɛnyɛ.', meaning: 'It is not good.' },
  { id: 'ScriptA_23', type: 'scripted', text: 'Ɛwɔ mu anaa?', meaning: 'Is it in stock?' },
  { id: 'ScriptA_24', type: 'scripted', text: 'Boa me!', meaning: 'Help me!' },
  { id: 'ScriptA_25', type: 'scripted', text: 'Me pa wo kyɛw, kyerɛ me kwan.', meaning: 'Please, guide me.' },
  { id: 'ScriptA_26', type: 'scripted', text: 'Kafra.', meaning: 'Sorry / Excuse me.' },
  { id: 'ScriptA_27', type: 'scripted', text: 'Mesrɛ wo, merehwehwɛ.', meaning: 'Please, I am searching for.' },
  { id: 'ScriptA_28', type: 'scripted', text: 'Mesrɛ wo, kyerɛkyerɛ mu ma me.', meaning: 'Please, explain it to me.' },
  { id: 'ScriptA_29', type: 'scripted', text: 'Mepɛ sɛ mesesa m'adwenkyerɛ.', meaning: 'I want to change my selection.' },
  { id: 'ScriptA_30', type: 'scripted', text: 'Sɛ meyi nneɛma yi a, ɛbɛyɛ sɛn?', meaning: 'What if I remove these items?' },
  { id: 'ScriptA_31', type: 'scripted', text: 'Hwɛ biribi foforɔ ma me.', meaning: 'Show me something else.' },
  { id: 'ScriptA_32', type: 'scripted', text: 'Me werɛ afi me akwankyerɛ nsɛm.', meaning: 'I have forgotten my password.' },
  { id: 'ScriptA_33', type: 'scripted', text: 'Menhu deɛ merehwehwɛ.', meaning: 'I cannot find what I am looking for.' },
  { id: 'ScriptA_34', type: 'scripted', text: 'Ɛrentumi nkɔ me nkrataa mu.', meaning: 'It cannot be added to my cart.' },
  { id: 'ScriptA_35', type: 'scripted', text: 'Mente deɛ woreka no aseɛ.', meaning: 'I don\'t understand what you are saying.' },
  { id: 'ScriptA_36', type: 'scripted', text: 'Me te wo aseɛ.', meaning: 'I understand you.' },
  { id: 'ScriptA_37', type: 'scripted', text: 'Kyerɛkyerɛ mu bio.', meaning: 'Explain again.' },
  { id: 'ScriptA_38', type: 'scripted', text: 'Mepɛ sɛ mekasa kyerɛ obi.', meaning: 'I want to speak to someone.' },
  { id: 'ScriptA_39', type: 'scripted', text: 'Adɛn nti na ɛnyɛ adwuma?', meaning: 'Why is it not working?' },
  { id: 'ScriptA_40', type: 'scripted', text: 'Ɛhe na mehunu adetɔ mmuaeɛ?', meaning: 'Where can I find order history?' },
  { id: 'ScriptA_41', type: 'scripted', text: 'Wobɛtumi aboa me atɔ biribi anaa?', meaning: 'Can you help me purchase something?' }
]

const scriptB: ScriptPrompt[] = [
  { id: 'ScriptB_1', type: 'scripted', text: 'Hwee', meaning: 'Zero / Nothing' },
  { id: 'ScriptB_2', type: 'scripted', text: 'Baako', meaning: 'One' },
  { id: 'ScriptB_3', type: 'scripted', text: 'Mmienu', meaning: 'Two' },
  { id: 'ScriptB_4', type: 'scripted', text: 'Mmiɛnsa', meaning: 'Three' },
  { id: 'ScriptB_5', type: 'scripted', text: 'Ɛnan', meaning: 'Four' },
  { id: 'ScriptB_6', type: 'scripted', text: 'Enum', meaning: 'Five' },
  { id: 'ScriptB_7', type: 'scripted', text: 'Nsia', meaning: 'Six' },
  { id: 'ScriptB_8', type: 'scripted', text: 'Nson', meaning: 'Seven' },
  { id: 'ScriptB_9', type: 'scripted', text: 'Nwɔtwe', meaning: 'Eight' },
  { id: 'ScriptB_10', type: 'scripted', text: 'Nkron', meaning: 'Nine' },
  { id: 'ScriptB_11', type: 'scripted', text: 'Edu', meaning: 'Ten' },
  { id: 'ScriptB_12', type: 'scripted', text: 'Mepɛ baako pɛ.', meaning: 'I want only one.' },
  { id: 'ScriptB_13', type: 'scripted', text: 'Mepɛ mmienu.', meaning: 'I want two.' },
  { id: 'ScriptB_14', type: 'scripted', text: 'Mepɛ mmiɛnsa.', meaning: 'I want three.' },
  { id: 'ScriptB_15', type: 'scripted', text: 'Atɔdeɛ no botaeɛ yɛ aduonu.', meaning: 'The order total is twenty.' },
  { id: 'ScriptB_16', type: 'scripted', text: 'Atɔdeɛ no bɛduru da aduonu baako.', meaning: 'The order will arrive on the twenty-first.' },
  { id: 'ScriptB_17', type: 'scripted', text: 'Mepɛ aduasa.', meaning: 'I want thirty.' },
  { id: 'ScriptB_18', type: 'scripted', text: 'Bra bio nna aduanan mu.', meaning: 'Come back in forty days.' },
  { id: 'ScriptB_19', type: 'scripted', text: 'Wɔbɛyi aduonum firi wo akantabuo mu.', meaning: 'Fifty will be deducted from your account.' },
  { id: 'ScriptB_20', type: 'scripted', text: 'Wɔbɛyi aduosia firi wo akantabuo mu.', meaning: 'Sixty will be deducted from your account.' },
  { id: 'ScriptB_21', type: 'scripted', text: 'Wɔbɛyi aduɔson firi wo akantabuo mu.', meaning: 'Seventy will be deducted from your account.' },
  { id: 'ScriptB_22', type: 'scripted', text: 'Wɔbɛyi aduowɔtwe firi wo akantabuo mu.', meaning: 'Eighty will be deducted from your account.' },
  { id: 'ScriptB_23', type: 'scripted', text: 'Wɔbɛyi aduokron firi wo akantabuo mu.', meaning: 'Ninety will be deducted from your account.' },
  { id: 'ScriptB_24', type: 'scripted', text: 'Ɛyɛ Ghana sidi ɔha.', meaning: 'It is one hundred Ghana Cedis.' },
  { id: 'ScriptB_25', type: 'scripted', text: 'Ɛyɛ Ghana sidi apem.', meaning: 'It is one thousand Ghana Cedis.' },
  { id: 'ScriptB_26', type: 'scripted', text: 'Ɛyɛ Ghana sidi ɔpepem.', meaning: 'It is one million Ghana Cedis.' },
  { id: 'ScriptB_27', type: 'scripted', text: 'Wɔbɛkra adwadeɛ no dɔnhwere mmienu mu.', meaning: 'The product will be delivered in two hours.' },
  { id: 'ScriptB_28', type: 'scripted', text: 'Store no bɛto nnɔnkron.', meaning: 'The store closes at nine o\'clock.' },
  { id: 'ScriptB_29', type: 'scripted', text: 'Sika a wɔbɛyi firi wo akantabuo mu ɛnnɛ yɛ Benada.', meaning: 'The withdrawal date is Tuesday.' },
  { id: 'ScriptB_30', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Ɛdwoada.', meaning: 'You will get your product on Monday.' },
  { id: 'ScriptB_31', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Ɛbenada.', meaning: 'You will get your product on Tuesday.' },
  { id: 'ScriptB_32', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Wukuada.', meaning: 'You will get your product on Wednesday.' },
  { id: 'ScriptB_33', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Yawoada.', meaning: 'You will get your product on Thursday.' },
  { id: 'ScriptB_34', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Fiada.', meaning: 'You will get your product on Friday.' },
  { id: 'ScriptB_35', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Memeneda.', meaning: 'You will get your product on Saturday.' },
  { id: 'ScriptB_36', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Kwasiada.', meaning: 'You will get your product on Sunday.' },
  { id: 'ScriptB_37', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔpɛpɔn.', meaning: 'I want my purchase history from January.' },
  { id: 'ScriptB_38', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔgyefoɔ.', meaning: 'I want my purchase history from February.' },
  { id: 'ScriptB_39', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔbɛnem.', meaning: 'I want my purchase history from March.' },
  { id: 'ScriptB_40', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Oforisuo.', meaning: 'I want my purchase history from April.' },
  { id: 'ScriptB_41', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Kotonimma.', meaning: 'I want my purchase history from May.' },
  { id: 'ScriptB_42', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ayɛwohomumɔ.', meaning: 'I want my purchase history from June.' },
  { id: 'ScriptB_43', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Kitawonsa.', meaning: 'I want my purchase history from July.' },
  { id: 'ScriptB_44', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔsanaa.', meaning: 'I want my purchase history from August.' },
  { id: 'ScriptB_45', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɛbɔ.', meaning: 'I want my purchase history from September.' },
  { id: 'ScriptB_46', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ahinime.', meaning: 'I want my purchase history from October.' },
  { id: 'ScriptB_47', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Obubuo.', meaning: 'I want my purchase history from November.' },
  { id: 'ScriptB_48', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔpɛnimma.', meaning: 'I want my purchase history from December.' },
  { id: 'ScriptB_49', type: 'scripted', text: 'Adwadeɛ yi boɔ yɛ sɛn?', meaning: 'How much is this product?' },
  { id: 'ScriptB_50', type: 'scripted', text: 'Ɛyɛ Ghana sidi aduonum.', meaning: 'It is fifty Ghana Cedis.' },
  { id: 'ScriptB_51', type: 'scripted', text: 'Mɛtua sidi ɔha ne aduasa enum.', meaning: 'I will pay one hundred and thirty-five cedis.' },

]

const scriptC: ScriptPrompt[] = [
  { id: 'ScriptC_1', type: 'scripted', text: 'Mepaakyɛw, telefon yi boɔ yɛ sɛn?', meaning: 'Please, how much is this phone?' },
  { id: 'ScriptC_2', type: 'scripted', text: 'Mepaakyɛw, laptop yi boɔ yɛ sɛn?', meaning: 'Please, how much is this laptop?' },
  { id: 'ScriptC_3', type: 'scripted', text: 'Te boɔ no so kakra ma me.', meaning: 'Give me a discount on the price.' },
  { id: 'ScriptC_4', type: 'scripted', text: 'Ntoma mmiɛnsa yi bɛyɛ sɛn?', meaning: 'How much for these three cloths?' },
  { id: 'ScriptC_5', type: 'scripted', text: 'Ma me adwadeɛ yi mmienu.', meaning: 'Give me two of this product.' },
  { id: 'ScriptC_6', type: 'scripted', text: 'Mode no bɛbrɛ me ɛwɔ fie anaa?', meaning: 'Will you deliver it to my home?' },
  { id: 'ScriptC_7', type: 'scripted', text: 'Merehwehwɛ mfonini hwɛdeɛ atɔ.', meaning: 'I am looking for a television to buy.' },
  { id: 'ScriptC_8', type: 'scripted', text: 'Fa wei kɔhyɛ me nkrataa mu.', meaning: 'Add this to my cart.' },
  { id: 'ScriptC_9', type: 'scripted', text: 'Mepaakyɛw, fa adwadeɛ no brɛ me ntɛm.', meaning: 'Please, deliver the product to me quickly.' },
  { id: 'ScriptC_10', type: 'scripted', text: 'Adwadeɛ foforɔ bɛn na wowɔ nnɛ?', meaning: 'What new products do you have today?' },
  { id: 'ScriptC_11', type: 'scripted', text: 'Me pɛ telefon a ne boɔ nyɛ den.', meaning: 'I want a phone that is not expensive.' },
  { id: 'ScriptC_12', type: 'scripted', text: 'Wei yɛ den dodo.', meaning: 'This is too expensive.' },
  { id: 'ScriptC_13', type: 'scripted', text: 'Ma me adwadeɛ a ɛwɔ mfoni papa.', meaning: 'Give me a product with good pictures.' },
  { id: 'ScriptC_14', type: 'scripted', text: 'Yɛbɛtumi ayi sika afiri akonhoma so?', meaning: 'Can we make a withdrawal from the card?' },
  { id: 'ScriptC_15', type: 'scripted', text: 'Wobɛtumi ayi sika firi me akantabuo mu ama me?', meaning: 'Can you process a refund for me?' },
  { id: 'ScriptC_16', type: 'scripted', text: 'M\'adwadeɛ no anyɛ fɛ.', meaning: 'The product is not good.' },
  { id: 'ScriptC_17', type: 'scripted', text: 'Adwadeɛ a wɔde brɛɛ me no ayɛ basabasa.', meaning: 'The delivered product is damaged.' },
  { id: 'ScriptC_18', type: 'scripted', text: 'Me pɛ sɛ mesesa adwadeɛ yi.', meaning: 'I want to exchange this product.' },
  { id: 'ScriptC_19', type: 'scripted', text: 'Adwadeɛ no nyɛ adwuma firi nnora.', meaning: 'The product hasn\'t been working since yesterday.' },
  { id: 'ScriptC_20', type: 'scripted', text: 'Ɛhe na yɛbɛnya mmoa wɔ adwuma no ho?', meaning: 'Where can we get help with this product?' },
  { id: 'ScriptC_21', type: 'scripted', text: 'Me hia teknikal mmoa ntɛm ara!', meaning: 'I need technical support immediately!' },
  { id: 'ScriptC_22', type: 'scripted', text: 'Me hia sɛ megye adwadeɛ yi to hɔ.', meaning: 'I need to return this product.' },
  { id: 'ScriptC_23', type: 'scripted', text: 'M\'atwerɛ ahyɛ m\'afutu a ɛdi kan.', meaning: 'I\'ve submitted my first review.' },
  { id: 'ScriptC_24', type: 'scripted', text: 'Mere hwɛ adwadeɛ no mu.', meaning: 'I am checking the product details.' },
  { id: 'ScriptC_25', type: 'scripted', text: 'Me pɛ sɛ me tɔ wei wɔ nnɔn mmienu mu.', meaning: 'I want to purchase this within two hours.' },
  { id: 'ScriptC_26', type: 'scripted', text: 'Mɛtumi de mobile money atua ka?', meaning: 'Can I pay using mobile money?' },
  { id: 'ScriptC_27', type: 'scripted', text: 'Mesrɛ wo, ma me receipt no bi.', meaning: 'Please, give me the receipt.' },
  { id: 'ScriptC_28', type: 'scripted', text: 'Merekɔ ahwɛ adwadeɛ foforɔ no.', meaning: 'I am going to check out the new products.' },
  { id: 'ScriptC_29', type: 'scripted', text: 'Kwan bɛn so na metumi ahwɛ m\'adetɔ ahyɛnsodeɛ?', meaning: 'How can I track my order?' },
  { id: 'ScriptC_30', type: 'scripted', text: 'Hwɛ adwadeɛ a ɛyɛ fɛ ma me.', meaning: 'Show me attractive products.' },
  { id: 'ScriptC_31', type: 'scripted', text: 'M\'ani gye adwadeɛ yi ho paa.', meaning: 'I really like this product.' },
  { id: 'ScriptC_32', type: 'scripted', text: 'Sika a wɔbɛgye ama adwadeɛ no nya krakra yɛ sɛn?', meaning: 'How much is the shipping fee?' },
  { id: 'ScriptC_33', type: 'scripted', text: 'M\'adwadeɛ no adu?', meaning: 'Has my order arrived?' },
  { id: 'ScriptC_34', type: 'scripted', text: 'M\'adwadeɛ no abɛn?', meaning: 'Is my order close to arriving?' },
]
const scriptD: ScriptPrompt[] = [
  { id: 'ScriptD_1', type: 'scripted', text: 'Sɛ wo akonhoma no nni sika a, yɛntumi ntɔ adwadeɛ no.', meaning: 'If your card has no money, we cannot purchase the product.' },
  { id: 'ScriptD_2', type: 'scripted', text: 'Mekɔɔ intanɛte no so nanso manhunu deɛ merepɛ.', meaning: 'I went online but I didn\'t find what I was looking for.' },
  { id: 'ScriptD_3', type: 'scripted', text: 'Adwadeɛ no asa enti merehwehwɛ bi foforɔ.', meaning: 'The product is out of stock so I am looking for another one.' },
  { id: 'ScriptD_4', type: 'scripted', text: 'Ɔyɛɛ account no efiri sɛ ɔpɛ sɛ ɔtɔ adwadeɛ no.', meaning: 'He/She created an account because he/she wants to buy the product.' },
  { id: 'ScriptD_5', type: 'scripted', text: 'Adwadeɛ a matɔ no nyinaa wɔ me nkrataa mu.', meaning: 'All the products I purchased are in my cart.' },
  { id: 'ScriptD_6', type: 'scripted', text: 'Wɔkaeɛ sɛ wɔbɛkra adwadeɛ no ɔkyena anɔpa.', meaning: 'They said that they will deliver the product tomorrow morning.' },
  { id: 'ScriptD_7', type: 'scripted', text: 'Wo akonhoma no da so te ase anaa?', meaning: 'Is your card still active?' },
  { id: 'ScriptD_8', type: 'scripted', text: 'Wo wɔ adwadeɛ sɛn wɔ wo nkrataa mu?', meaning: 'How many products do you have in your cart?' },
  { id: 'ScriptD_9', type: 'scripted', text: 'Me akonhoma no yɛ credit card.', meaning: 'My card is a credit card.' },
  { id: 'ScriptD_10', type: 'scripted', text: 'Online adetɔ dwumadi no na ɛboa me berɛ a mehia adwadeɛ.', meaning: 'The online shopping platform is what helps me when I need products.' },
  { id: 'ScriptD_11', type: 'scripted', text: 'M\'ani agye sɛ m\'adwadeɛ no aba nnɛ.', meaning: 'I am happy that my product has arrived today.' },
  { id: 'ScriptD_12', type: 'scripted', text: 'Akonhoma tumi yɛ no yɛ den nanso ɛho hia.', meaning: 'Managing cards is difficult but it is important.' },
  { id: 'ScriptD_13', type: 'scripted', text: 'Mensusu sɛ wɔbɛtumi akra adwadeɛ no nnɛ.', meaning: 'I don\'t think they can deliver the product today.' },
  { id: 'ScriptD_14', type: 'scripted', text: 'Ɛwɔ sɛ yɛ di adetɔ mmarasɛm so.', meaning: 'We must follow the shopping rules.' },
  { id: 'ScriptD_15', type: 'scripted', text: 'Me werɛ aho sɛ adwadeɛ a metɔeɛ no anyɛ adwuma.', meaning: 'I am very sad that the product I bought is not working.' },
  { id: 'ScriptD_16', type: 'scripted', text: 'M\'abrɛ paa sɛ merehwehwɛ adwadeɛ pa.', meaning: 'I am very tired of searching for quality products.' },
  { id: 'ScriptD_17', type: 'scripted', text: 'Customer service no pɛ sɛ wɔne wo kasa.', meaning: 'Customer service wants to talk to you.' },
  { id: 'ScriptD_18', type: 'scripted', text: 'Mepɛ sɛ mesua online adetɔ yie.', meaning: 'I want to learn online shopping well.' },
  { id: 'ScriptD_19', type: 'scripted', text: 'Kenkan adwadeɛ no ho nsɛm kyerɛ me.', meaning: 'Read the product description to me.' },
  { id: 'ScriptD_20', type: 'scripted', text: 'Online store no so paa na ɛwɔ intanɛte so.', meaning: 'The online store is very big and it is on the internet.' },
  { id: 'ScriptD_21', type: 'scripted', text: 'Adwadeɛ tenten bi si mfonini no mu.', meaning: 'A tall product is in the image.' },
  { id: 'ScriptD_22', type: 'scripted', text: 'Adɛn nti na adwadeɛ no akra kyɛeɛ?', meaning: 'Why was the product delivery late?' },
  { id: 'ScriptD_23', type: 'scripted', text: 'Ɛbɛyɛ nnɔn sɛn ansa na woawie adetɔ no?', meaning: 'What time will it take before you finish the purchase?' },
  { id: 'ScriptD_24', type: 'scripted', text: 'Wo ne hwan na moretɔ adwadeɛ no?', meaning: 'With whom are you buying the product?' },
  { id: 'ScriptD_25', type: 'scripted', text: 'Mpɛn ahe na wo tɔ adwadeɛ firi online?', meaning: 'How many times do you buy products online?' },
  { id: 'ScriptD_26', type: 'scripted', text: 'Nnora, mekɔɔ intanɛte adetɔ dwumadi no so.', meaning: 'Yesterday, I visited the online shopping platform.' },
  { id: 'ScriptD_27', type: 'scripted', text: 'Ɔkyena wɔbɛkra adwadeɛ no aba.', meaning: 'Tomorrow they will deliver the product.' },
  { id: 'ScriptD_28', type: 'scripted', text: 'Berɛ a na meretɔ adwadeɛ no, me nua no baeɛ.', meaning: 'While I was purchasing the product, my sibling came.' },
  { id: 'ScriptD_29', type: 'scripted', text: 'Ɔtumi di adetɔ dwumadi no ho dwuma fɛfɛɛfɛ.', meaning: 'He/She can use the shopping platform beautifully/efficiently.' },
]
const spontaneousPrompts: ScriptPrompt[] = [
  { id: 'Spontaneous_1', type: 'spontaneous', text: 'Introduce Yourself – Name, age, where you live, what you do. (Mesrɛ wo, ka wo ho asɛm kyerɛ me. Wo din, w’adi mfeɛ sɛn, ɛhe na wote, adwuma bɛn na wokɔ?)' },
  { id: 'Spontaneous_2', type: 'spontaneous', text: 'Daily Routine – How you start and end your day. (Kyerɛ me sɛnea wo da no kɔeɛ fiti anɔpa kɔpem anadwo)' },
  { id: 'Spontaneous_3', type: 'spontaneous', text: 'Describe Your Environment – Talk about what you see around you right now. (Kasa fa nnoɔma a ɛwɔ baabi a wote seesei yi ho.)' },
  { id: 'Spontaneous_4', type: 'spontaneous', text: 'Ka asɛm bi a ɛsi nnansa yi ara a ɛmaa w\'ani gyeeɛ paa anaasɛ ɛmaa wo werɛ hoeɛ ho asɛm kyerɛ me.Adɛn nti na ɛte saa?. (Tell me about something that happened recently that made you very happy or very sad.Why did it make you feel that way ?)' },
  { id: 'Spontaneous_5', type: 'spontaneous', text: 'Kyerɛkyerɛ me sɛnea yɛnoa Twi aduane a wopɛ pa ara no fiase kɔpem awieeɛ. Nnoɔma bɛn na wohia?. (Explain to me how your favorite Twi food is prepared from start to finish.What ingredients do you need?)' },
  { id: 'Spontaneous_6', type: 'spontaneous', text: 'Hobbies & Interests – What do you like to do in your free time? (Dɛn na wopɛ sɛ woyɛ wɔ wo bere a w\'anya fa no mu?)' },
  { id: 'Spontaneous_7', type: 'spontaneous', text: 'Faako a woteɛ no, kyerɛkyerɛ akwantuo a wofa firi wo fie kɔ w\'adwumam anaa sukuu mu. (Where you live, describe the journey you take from your home to your workplace or school.)' },
  { id: 'Spontaneous_8', type: 'spontaneous', text: 'Sɛ wohyia ɔhaw bi wɔ w\'asetenam a, hwan na odi kan boa wo, na ɛkwan bɛn so?. (If you face a problem in your life, who helps you first, and in what way ?)' }
]

// --- Define Sections ---
export const RECORDING_SECTIONS: RecordingSection[] = [
  {
    id: 'ScriptA',
    title: 'Common Phrases',
    description: 'Read the following common Twi phrases aloud clearly.',
    prompts: scriptA,
  },
  {
    id: 'ScriptB',
    title: 'Numbers, Dates & Time',
    description: 'Read the following numbers, days, months, and time-related phrases.',
    prompts: scriptB,
  },
  {
    id: 'ScriptC',
    title: 'Everyday Situations',
    description: 'Read phrases related to shopping, food, health, work, and travel.',
    prompts: scriptC,
  },
  {
    id: 'ScriptD',
    title: 'More Conversation',
    description: 'Read these slightly more complex sentences and questions.',
    prompts: scriptD,
  },
  {
    id: 'Spontaneous',
    title: 'Spontaneous Speech',
    description: 'For the next prompts, read the topic, then speak freely and naturally about it in Twi for 30-60 seconds. DO NOT read the instructions aloud.',
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
export const SPONTANEOUS_PROMPTS_COUNT = 8

export const getGlobalPromptIndex = (sectionIndex: number, promptInSectionIndex: number): number => {
  let globalIndex = 0;
  for (let i = 0; i < sectionIndex; i++) {
    globalIndex += RECORDING_SECTIONS[i].prompts.length;
  }
  globalIndex += promptInSectionIndex;
  return globalIndex;
};
