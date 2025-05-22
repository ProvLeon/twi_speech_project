import { RecordingSection, ScriptPrompt } from '@/types';

const scriptA: ScriptPrompt[] = [
  { id: 'ScriptAU_1', type: 'scripted', text: 'Maakye', meaning: 'Good morning' },
  { id: 'ScriptAU_2', type: 'scripted', text: 'Maaha', meaning: 'Good afternoon' },
  { id: 'ScriptAU_3', type: 'scripted', text: 'Maadwo', meaning: 'Good evening/night' },
  { id: 'ScriptAU_4', type: 'scripted', text: 'Wo ho te sɛn?', meaning: 'How are you?' },
  { id: 'ScriptAU_5', type: 'scripted', text: 'Me ho yɛ.', meaning: 'I am fine.' },
  { id: 'ScriptAU_6', type: 'scripted', text: 'Bɔkɔɔ.', meaning: 'Fine / Cool / Okay.' },
  { id: 'ScriptAU_7', type: 'scripted', text: 'Medaase', meaning: 'Thank you.' },
  { id: 'ScriptAU_8', type: 'scripted', text: 'Medaase paa.', meaning: 'Thank you very much.' },
  { id: 'ScriptAU_9', type: 'scripted', text: 'Mepɛ sɛ metɔ biribi.', meaning: 'I want to buy something.' },
  { id: 'ScriptAU_10', type: 'scripted', text: 'Metumi akɔtwe sika?', meaning: 'Can I withdraw money?' },
  { id: 'ScriptAU_11', type: 'scripted', text: 'Mesrɛ wo, bue me akantabuo.', meaning: 'Please, open an account for me.' },
  { id: 'ScriptAU_12', type: 'scripted', text: 'Hwehwɛ atɔdeɛ.', meaning: 'Search for products.' },
  { id: 'ScriptAU_13', type: 'scripted', text: 'Fa kɔ nkrataa mu.', meaning: 'Go to cart.' },
  { id: 'ScriptAU_14', type: 'scripted', text: 'Kɔ w\'anim.', meaning: 'Go forward/ Next.' },
  { id: 'ScriptAU_15', type: 'scripted', text: 'San w\'akyi.', meaning: 'Go back/ Previous.' },
  { id: 'ScriptAU_16', type: 'scripted', text: 'Fa gu me nkrataa mu.', meaning: 'Add to my cart.' },
  { id: 'ScriptAU_17', type: 'scripted', text: 'Yi firi nkrataa mu.', meaning: 'Remove from cart.' },
  { id: 'ScriptAU_18', type: 'scripted', text: 'Me pɛ sɛ mewie tɔneɛ.', meaning: 'I want to checkout.' },
  { id: 'ScriptAU_19', type: 'scripted', text: 'Sɛn na metumi atua ka?', meaning: 'How can I pay?' },
  { id: 'ScriptAU_20', type: 'scripted', text: 'Wɔde bɛkra me ɛhe?', meaning: 'Where will it be delivered?' },
  { id: 'ScriptAU_21', type: 'scripted', text: 'Ɛyɛ.', meaning: 'It is good / Okay.' },
  { id: 'ScriptAU_22', type: 'scripted', text: 'Ɛnyɛ.', meaning: 'It is not good.' },
  { id: 'ScriptAU_23', type: 'scripted', text: 'Ɛwɔ mu anaa?', meaning: 'Is it in stock?' },
  { id: 'ScriptAU_24', type: 'scripted', text: 'Boa me!', meaning: 'Help me!' },
  { id: 'ScriptAU_25', type: 'scripted', text: 'Me pa wo kyɛw, kyerɛ me kwan.', meaning: 'Please, guide me.' },
  { id: 'ScriptAU_26', type: 'scripted', text: 'Kafra.', meaning: 'Sorry / Excuse me.' },
  { id: 'ScriptAU_27', type: 'scripted', text: 'Mesrɛ wo, merehwehwɛ.', meaning: 'Please, I am searching for.' },
  { id: 'ScriptAU_28', type: 'scripted', text: 'Mesrɛ wo, kyerɛkyerɛ mu ma me.', meaning: 'Please, explain it to me.' },
  {
    id: 'ScriptAU_29', type: 'scripted', text: 'Mepɛ sɛ mesesa m\'adwenkyerɛ.', meaning: 'I want to change my selection.'
  },
  { id: 'ScriptAU_30', type: 'scripted', text: 'Sɛ meyi nneɛma yi a, ɛbɛyɛ sɛn?', meaning: 'What if I remove these items?' },
  { id: 'ScriptAU_31', type: 'scripted', text: 'Hwɛ biribi foforɔ ma me.', meaning: 'Show me something else.' },
  { id: 'ScriptAU_32', type: 'scripted', text: 'Me werɛ afi me akwankyerɛ nsɛm.', meaning: 'I have forgotten my password.' },
  { id: 'ScriptAU_33', type: 'scripted', text: 'Menhu deɛ merehwehwɛ.', meaning: 'I cannot find what I am looking for.' },
  { id: 'ScriptAU_34', type: 'scripted', text: 'Ɛrentumi nkɔ me nkrataa mu.', meaning: 'It cannot be added to my cart.' },
  { id: 'ScriptAU_35', type: 'scripted', text: 'Mente deɛ woreka no aseɛ.', meaning: 'I don\'t understand what you are saying.' },
  { id: 'ScriptAU_36', type: 'scripted', text: 'Me te wo aseɛ.', meaning: 'I understand you.' },
  { id: 'ScriptAU_37', type: 'scripted', text: 'Kyerɛkyerɛ mu bio.', meaning: 'Explain again.' },
  { id: 'ScriptAU_38', type: 'scripted', text: 'Mepɛ sɛ mekasa kyerɛ obi.', meaning: 'I want to speak to someone.' },
  { id: 'ScriptAU_39', type: 'scripted', text: 'Adɛn nti na ɛnyɛ adwuma?', meaning: 'Why is it not working?' },
  { id: 'ScriptAU_40', type: 'scripted', text: 'Ɛhe na mehunu adetɔ mmuaeɛ?', meaning: 'Where can I find order history?' },
  { id: 'ScriptAU_41', type: 'scripted', text: 'Wobɛtumi aboa me atɔ biribi anaa?', meaning: 'Can you help me purchase something?' }
]

const scriptB: ScriptPrompt[] = [
  { id: 'ScriptBU_1', type: 'scripted', text: 'Hwee', meaning: 'Zero / Nothing' },
  { id: 'ScriptBU_2', type: 'scripted', text: 'Baako', meaning: 'One' },
  { id: 'ScriptBU_3', type: 'scripted', text: 'Mmienu', meaning: 'Two' },
  { id: 'ScriptBU_4', type: 'scripted', text: 'Mmiɛnsa', meaning: 'Three' },
  { id: 'ScriptBU_5', type: 'scripted', text: 'Ɛnan', meaning: 'Four' },
  { id: 'ScriptBU_6', type: 'scripted', text: 'Enum', meaning: 'Five' },
  { id: 'ScriptBU_7', type: 'scripted', text: 'Nsia', meaning: 'Six' },
  { id: 'ScriptBU_8', type: 'scripted', text: 'Nson', meaning: 'Seven' },
  { id: 'ScriptBU_9', type: 'scripted', text: 'Nwɔtwe', meaning: 'Eight' },
  { id: 'ScriptBU_10', type: 'scripted', text: 'Nkron', meaning: 'Nine' },
  { id: 'ScriptBU_11', type: 'scripted', text: 'Edu', meaning: 'Ten' },
  { id: 'ScriptBU_12', type: 'scripted', text: 'Mepɛ baako pɛ.', meaning: 'I want only one.' },
  { id: 'ScriptBU_13', type: 'scripted', text: 'Mepɛ mmienu.', meaning: 'I want two.' },
  { id: 'ScriptBU_14', type: 'scripted', text: 'Mepɛ mmiɛnsa.', meaning: 'I want three.' },
  { id: 'ScriptBU_15', type: 'scripted', text: 'Atɔdeɛ no botaeɛ yɛ aduonu.', meaning: 'The order total is twenty.' },
  { id: 'ScriptBU_16', type: 'scripted', text: 'Atɔdeɛ no bɛduru da aduonu baako.', meaning: 'The order will arrive on the twenty-first.' },
  { id: 'ScriptBU_17', type: 'scripted', text: 'Mepɛ aduasa.', meaning: 'I want thirty.' },
  { id: 'ScriptBU_18', type: 'scripted', text: 'Bra bio nna aduanan mu.', meaning: 'Come back in forty days.' },
  { id: 'ScriptBU_19', type: 'scripted', text: 'Wɔbɛyi aduonum firi wo akantabuo mu.', meaning: 'Fifty will be deducted from your account.' },
  { id: 'ScriptBU_20', type: 'scripted', text: 'Wɔbɛyi aduosia firi wo akantabuo mu.', meaning: 'Sixty will be deducted from your account.' },
  { id: 'ScriptBU_21', type: 'scripted', text: 'Wɔbɛyi aduɔson firi wo akantabuo mu.', meaning: 'Seventy will be deducted from your account.' },
  { id: 'ScriptBU_22', type: 'scripted', text: 'Wɔbɛyi aduowɔtwe firi wo akantabuo mu.', meaning: 'Eighty will be deducted from your account.' },
  { id: 'ScriptBU_23', type: 'scripted', text: 'Wɔbɛyi aduokron firi wo akantabuo mu.', meaning: 'Ninety will be deducted from your account.' },
  { id: 'ScriptBU_24', type: 'scripted', text: 'Ɛyɛ Ghana sidi ɔha.', meaning: 'It is one hundred Ghana Cedis.' },
  { id: 'ScriptBU_25', type: 'scripted', text: 'Ɛyɛ Ghana sidi apem.', meaning: 'It is one thousand Ghana Cedis.' },
  { id: 'ScriptBU_26', type: 'scripted', text: 'Ɛyɛ Ghana sidi ɔpepem.', meaning: 'It is one million Ghana Cedis.' },
  { id: 'ScriptBU_27', type: 'scripted', text: 'Wɔbɛkra adwadeɛ no dɔnhwere mmienu mu.', meaning: 'The product will be delivered in two hours.' },
  { id: 'ScriptBU_28', type: 'scripted', text: 'Store no bɛto nnɔnkron.', meaning: 'The store closes at nine o\'clock.' },
  { id: 'ScriptBU_29', type: 'scripted', text: 'Sika a wɔbɛyi firi wo akantabuo mu ɛnnɛ yɛ Benada.', meaning: 'The withdrawal date is Tuesday.' },
  { id: 'ScriptBU_30', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Ɛdwoada.', meaning: 'You will get your product on Monday.' },
  { id: 'ScriptBU_31', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Ɛbenada.', meaning: 'You will get your product on Tuesday.' },
  { id: 'ScriptBU_32', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Wukuada.', meaning: 'You will get your product on Wednesday.' },
  { id: 'ScriptBU_33', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Yawoada.', meaning: 'You will get your product on Thursday.' },
  { id: 'ScriptBU_34', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Fiada.', meaning: 'You will get your product on Friday.' },
  { id: 'ScriptBU_35', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Memeneda.', meaning: 'You will get your product on Saturday.' },
  { id: 'ScriptBU_36', type: 'scripted', text: 'Wobɛnya wo adwadeɛ no Kwasiada.', meaning: 'You will get your product on Sunday.' },
  { id: 'ScriptBU_37', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔpɛpɔn.', meaning: 'I want my purchase history from January.' },
  { id: 'ScriptBU_38', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔgyefoɔ.', meaning: 'I want my purchase history from February.' },
  { id: 'ScriptBU_39', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔbɛnem.', meaning: 'I want my purchase history from March.' },
  { id: 'ScriptBU_40', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Oforisuo.', meaning: 'I want my purchase history from April.' },
  { id: 'ScriptBU_41', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Kotonimma.', meaning: 'I want my purchase history from May.' },
  { id: 'ScriptBU_42', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ayɛwohomumɔ.', meaning: 'I want my purchase history from June.' },
  { id: 'ScriptBU_43', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Kitawonsa.', meaning: 'I want my purchase history from July.' },
  { id: 'ScriptBU_44', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔsanaa.', meaning: 'I want my purchase history from August.' },
  { id: 'ScriptBU_45', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɛbɔ.', meaning: 'I want my purchase history from September.' },
  { id: 'ScriptBU_46', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ahinime.', meaning: 'I want my purchase history from October.' },
  { id: 'ScriptBU_47', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Obubuo.', meaning: 'I want my purchase history from November.' },
  { id: 'ScriptBU_48', type: 'scripted', text: 'Mɛpɛ me adetɔ nsɛm firi Ɔpɛnimma.', meaning: 'I want my purchase history from December.' },
  { id: 'ScriptBU_49', type: 'scripted', text: 'Adwadeɛ yi boɔ yɛ sɛn?', meaning: 'How much is this product?' },
  { id: 'ScriptBU_50', type: 'scripted', text: 'Ɛyɛ Ghana sidi aduonum.', meaning: 'It is fifty Ghana Cedis.' },
  { id: 'ScriptBU_51', type: 'scripted', text: 'Mɛtua sidi ɔha ne aduasa enum.', meaning: 'I will pay one hundred and thirty-five cedis.' },

]

const scriptC: ScriptPrompt[] = [
  { id: 'ScriptCU_1', type: 'scripted', text: 'Mepaakyɛw, telefon yi boɔ yɛ sɛn?', meaning: 'Please, how much is this phone?' },
  { id: 'ScriptCU_2', type: 'scripted', text: 'Mepaakyɛw, laptop yi boɔ yɛ sɛn?', meaning: 'Please, how much is this laptop?' },
  { id: 'ScriptCU_3', type: 'scripted', text: 'Te boɔ no so kakra ma me.', meaning: 'Give me a discount on the price.' },
  { id: 'ScriptCU_4', type: 'scripted', text: 'Ntoma mmiɛnsa yi bɛyɛ sɛn?', meaning: 'How much for these three cloths?' },
  { id: 'ScriptCU_5', type: 'scripted', text: 'Ma me adwadeɛ yi mmienu.', meaning: 'Give me two of this product.' },
  { id: 'ScriptCU_6', type: 'scripted', text: 'Mode no bɛbrɛ me ɛwɔ fie anaa?', meaning: 'Will you deliver it to my home?' },
  { id: 'ScriptCU_7', type: 'scripted', text: 'Merehwehwɛ mfonini hwɛdeɛ atɔ.', meaning: 'I am looking for a television to buy.' },
  { id: 'ScriptCU_8', type: 'scripted', text: 'Fa wei kɔhyɛ me nkrataa mu.', meaning: 'Add this to my cart.' },
  { id: 'ScriptCU_9', type: 'scripted', text: 'Mepaakyɛw, fa adwadeɛ no brɛ me ntɛm.', meaning: 'Please, deliver the product to me quickly.' },
  { id: 'ScriptCU_10', type: 'scripted', text: 'Adwadeɛ foforɔ bɛn na wowɔ nnɛ?', meaning: 'What new products do you have today?' },
  { id: 'ScriptCU_11', type: 'scripted', text: 'Me pɛ telefon a ne boɔ nyɛ den.', meaning: 'I want a phone that is not expensive.' },
  { id: 'ScriptCU_12', type: 'scripted', text: 'Wei yɛ den dodo.', meaning: 'This is too expensive.' },
  { id: 'ScriptCU_13', type: 'scripted', text: 'Ma me adwadeɛ a ɛwɔ mfoni papa.', meaning: 'Give me a product with good pictures.' },
  { id: 'ScriptCU_14', type: 'scripted', text: 'Yɛbɛtumi ayi sika afiri akonhoma so?', meaning: 'Can we make a withdrawal from the card?' },
  { id: 'ScriptCU_15', type: 'scripted', text: 'Wobɛtumi ayi sika firi me akantabuo mu ama me?', meaning: 'Can you process a refund for me?' },
  {
    id: 'ScriptCU_16', type: 'scripted', text: 'M\'adwadeɛ no anyɛ fɛ.', meaning: 'The product is not good.'
  },
  { id: 'ScriptCU_17', type: 'scripted', text: 'Adwadeɛ a wɔde brɛɛ me no ayɛ basabasa.', meaning: 'The delivered product is damaged.' },
  { id: 'ScriptCU_18', type: 'scripted', text: 'Me pɛ sɛ mesesa adwadeɛ yi.', meaning: 'I want to exchange this product.' },
  { id: 'ScriptCU_19', type: 'scripted', text: 'Adwadeɛ no nyɛ adwuma firi nnora.', meaning: 'The product hasn\'t been working since yesterday.' },
  { id: 'ScriptCU_20', type: 'scripted', text: 'Ɛhe na yɛbɛnya mmoa wɔ adwuma no ho?', meaning: 'Where can we get help with this product?' },
  { id: 'ScriptCU_21', type: 'scripted', text: 'Me hia teknikal mmoa ntɛm ara!', meaning: 'I need technical support immediately!' },
  { id: 'ScriptCU_22', type: 'scripted', text: 'Me hia sɛ megye adwadeɛ yi to hɔ.', meaning: 'I need to return this product.' },
  { id: 'ScriptCU_23', type: 'scripted', text: 'M\'atwerɛ ahyɛ m\'afutu a ɛdi kan.', meaning: 'I\'ve submitted my first review.' },
  { id: 'ScriptCU_24', type: 'scripted', text: 'Mere hwɛ adwadeɛ no mu.', meaning: 'I am checking the product details.' },
  { id: 'ScriptCU_25', type: 'scripted', text: 'Me pɛ sɛ me tɔ wei wɔ nnɔn mmienu mu.', meaning: 'I want to purchase this within two hours.' },
  { id: 'ScriptCU_26', type: 'scripted', text: 'Mɛtumi de mobile money atua ka?', meaning: 'Can I pay using mobile money?' },
  { id: 'ScriptCU_27', type: 'scripted', text: 'Mesrɛ wo, ma me receipt no bi.', meaning: 'Please, give me the receipt.' },
  { id: 'ScriptCU_28', type: 'scripted', text: 'Merekɔ ahwɛ adwadeɛ foforɔ no.', meaning: 'I am going to check out the new products.' },
  {
    id: 'ScriptCU_29', type: 'scripted', text: 'Kwan bɛn so na metumi ahwɛ m\'adetɔ ahyɛnsodeɛ?', meaning: 'How can I track my order?'
  },
  { id: 'ScriptCU_30', type: 'scripted', text: 'Hwɛ adwadeɛ a ɛyɛ fɛ ma me.', meaning: 'Show me attractive products.' },
  { id: 'ScriptCU_31', type: 'scripted', text: 'M\'ani gye adwadeɛ yi ho paa.', meaning: 'I really like this product.' },
  { id: 'ScriptCU_32', type: 'scripted', text: 'Sika a wɔbɛgye ama adwadeɛ no nya krakra yɛ sɛn?', meaning: 'How much is the shipping fee?' },
  { id: 'ScriptCU_33', type: 'scripted', text: 'M\'adwadeɛ no adu?', meaning: 'Has my order arrived?' },
  { id: 'ScriptCU_34', type: 'scripted', text: 'M\'adwadeɛ no abɛn?', meaning: 'Is my order close to arriving?' },
]
const scriptD: ScriptPrompt[] = [
  { id: 'ScriptDU_1', type: 'scripted', text: 'Sɛ wo akonhoma no nni sika a, yɛntumi ntɔ adwadeɛ no.', meaning: 'If your card has no money, we cannot purchase the product.' },
  { id: 'ScriptDU_2', type: 'scripted', text: 'Mekɔɔ intanɛte no so nanso manhunu deɛ merepɛ.', meaning: 'I went online but I didn\'t find what I was looking for.' },
  { id: 'ScriptDU_3', type: 'scripted', text: 'Adwadeɛ no asa enti merehwehwɛ bi foforɔ.', meaning: 'The product is out of stock so I am looking for another one.' },
  { id: 'ScriptDU_4', type: 'scripted', text: 'Ɔyɛɛ account no efiri sɛ ɔpɛ sɛ ɔtɔ adwadeɛ no.', meaning: 'He/She created an account because he/she wants to buy the product.' },
  { id: 'ScriptDU_5', type: 'scripted', text: 'Adwadeɛ a matɔ no nyinaa wɔ me nkrataa mu.', meaning: 'All the products I purchased are in my cart.' },
  { id: 'ScriptDU_6', type: 'scripted', text: 'Wɔkaeɛ sɛ wɔbɛkra adwadeɛ no ɔkyena anɔpa.', meaning: 'They said that they will deliver the product tomorrow morning.' },
  { id: 'ScriptDU_7', type: 'scripted', text: 'Wo akonhoma no da so te ase anaa?', meaning: 'Is your card still active?' },
  { id: 'ScriptDU_8', type: 'scripted', text: 'Wo wɔ adwadeɛ sɛn wɔ wo nkrataa mu?', meaning: 'How many products do you have in your cart?' },
  { id: 'ScriptDU_9', type: 'scripted', text: 'Me akonhoma no yɛ credit card.', meaning: 'My card is a credit card.' },
  { id: 'ScriptDU_10', type: 'scripted', text: 'Online adetɔ dwumadi no na ɛboa me berɛ a mehia adwadeɛ.', meaning: 'The online shopping platform is what helps me when I need products.' },
  { id: 'ScriptDU_11', type: 'scripted', text: 'M\'ani agye sɛ m\'adwadeɛ no aba nnɛ.', meaning: 'I am happy that my product has arrived today.' },
  { id: 'ScriptDU_12', type: 'scripted', text: 'Akonhoma tumi yɛ no yɛ den nanso ɛho hia.', meaning: 'Managing cards is difficult but it is important.' },
  { id: 'ScriptDU_13', type: 'scripted', text: 'Mensusu sɛ wɔbɛtumi akra adwadeɛ no nnɛ.', meaning: 'I don\'t think they can deliver the product today.' },
  { id: 'ScriptDU_14', type: 'scripted', text: 'Ɛwɔ sɛ yɛ di adetɔ mmarasɛm so.', meaning: 'We must follow the shopping rules.' },
  { id: 'ScriptDU_15', type: 'scripted', text: 'Me werɛ aho sɛ adwadeɛ a metɔeɛ no anyɛ adwuma.', meaning: 'I am very sad that the product I bought is not working.' },
  { id: 'ScriptDU_16', type: 'scripted', text: 'M\'abrɛ paa sɛ merehwehwɛ adwadeɛ pa.', meaning: 'I am very tired of searching for quality products.' },
  { id: 'ScriptDU_17', type: 'scripted', text: 'Customer service no pɛ sɛ wɔne wo kasa.', meaning: 'Customer service wants to talk to you.' },
  { id: 'ScriptDU_18', type: 'scripted', text: 'Mepɛ sɛ mesua online adetɔ yie.', meaning: 'I want to learn online shopping well.' },
  { id: 'ScriptDU_19', type: 'scripted', text: 'Kenkan adwadeɛ no ho nsɛm kyerɛ me.', meaning: 'Read the product description to me.' },
  { id: 'ScriptDU_20', type: 'scripted', text: 'Online store no so paa na ɛwɔ intanɛte so.', meaning: 'The online store is very big and it is on the internet.' },
  { id: 'ScriptDU_21', type: 'scripted', text: 'Adwadeɛ tenten bi si mfonini no mu.', meaning: 'A tall product is in the image.' },
  { id: 'ScriptDU_22', type: 'scripted', text: 'Adɛn nti na adwadeɛ no akra kyɛeɛ?', meaning: 'Why was the product delivery late?' },
  { id: 'ScriptDU_23', type: 'scripted', text: 'Ɛbɛyɛ nnɔn sɛn ansa na woawie adetɔ no?', meaning: 'What time will it take before you finish the purchase?' },
  { id: 'ScriptDU_24', type: 'scripted', text: 'Wo ne hwan na moretɔ adwadeɛ no?', meaning: 'With whom are you buying the product?' },
  { id: 'ScriptDU_25', type: 'scripted', text: 'Mpɛn ahe na wo tɔ adwadeɛ firi online?', meaning: 'How many times do you buy products online?' },
  { id: 'ScriptDU_26', type: 'scripted', text: 'Nnora, mekɔɔ intanɛte adetɔ dwumadi no so.', meaning: 'Yesterday, I visited the online shopping platform.' },
  { id: 'ScriptDU_27', type: 'scripted', text: 'Ɔkyena wɔbɛkra adwadeɛ no aba.', meaning: 'Tomorrow they will deliver the product.' },
  { id: 'ScriptDU_28', type: 'scripted', text: 'Berɛ a na meretɔ adwadeɛ no, me nua no baeɛ.', meaning: 'While I was purchasing the product, my sibling came.' },
  { id: 'ScriptDU_29', type: 'scripted', text: 'Ɔtumi di adetɔ dwumadi no ho dwuma fɛfɛɛfɛ.', meaning: 'He/She can use the shopping platform beautifully/efficiently.' },
]
const spontaneousPrompts: ScriptPrompt[] = [
  { id: 'SpontaneousU_1', type: 'spontaneous', text: 'Online Shopping Experience – Talk about how you shop online, what websites or apps you use, and what products you usually buy. (Kyerɛkyerɛ sɛnea wotɔ nneɛma wɔ intanɛte so, websites anaa apps a wode di dwuma ne nneɛma a wotaa tɔ.)' },
  { id: 'SpontaneousU_2', type: 'spontaneous', text: 'Shopping Preferences – Describe your shopping habits. Do you prefer online or physical stores? Why? (Kyerɛkyerɛ w\'adetɔ suban. Wopɛ adetɔ wɔ intanɛte so anaa wopɛ sɛ wokɔ stores mu? Adɛn?)' },
  { id: 'SpontaneousU_3', type: 'spontaneous', text: 'Describe an online product you want to buy – Explain its features, price, and why you need it. (Kyerɛkyerɛ adwadeɛ bi a wopɛ sɛ wotɔ wɔ intanɛte so – Ka ne su, ne boɔ, ne adɛn nti a ɛho hia wo.)' },
  { id: 'SpontaneousU_4', type: 'spontaneous', text: 'Ka adetɔ anaa adwadeɛ bi a wotɔeɛ nnansa yi a ɛmaa w\'ani gyeeɛ paa anaasɛ ɛmaa wo werɛ hoeɛ ho asɛm kyerɛ me. Adɛn nti na ɛte saa? (Tell me about a recent purchase or product that made you very happy or disappointed. Why did it make you feel that way?)' },
  { id: 'SpontaneousU_5', type: 'spontaneous', text: 'Kyerɛkyerɛ kwan a wɔfa so tɔ biribi wɔ online shopping platform so fiase kɔpem awieeɛ. Nnoɔma bɛn ne nsɛnkyerɛnnee a wohia? (Explain the step-by-step process of making a purchase on an online shopping platform. What information and details do you need?)' },
  { id: 'SpontaneousU_6', type: 'spontaneous', text: 'Payment Methods – What payment methods do you prefer when shopping online and why? (Sika tua akwan bɛn na wopɛ berɛ a woretɔ adeɛ wɔ internet so, na adɛn?)' },
  { id: 'SpontaneousU_7', type: 'spontaneous', text: 'Your ideal e-commerce voice assistant – Describe what features and capabilities you would want in a voice assistant for shopping online. (Kyerɛkyerɛ su ne ahoɔden a wopɛ sɛ nkasa mmoa dwumadi a wɔde tɔ adeɛ wɔ internet so bɛwɔ.)' },
  { id: 'SpontaneousU_8', type: 'spontaneous', text: 'Sɛ wohyia ɔhaw bi wɔ online shopping mu a, dɛn na woyɛ? Kyerɛkyerɛ mmoa a wopɛ. (If you encounter a problem while shopping online, what do you do? Describe the kind of help you would want.)' }
]

// --- Define Sections ---
export const RECORDING_SECTIONS: RecordingSection[] = [
  {
    id: 'ScriptAU',
    title: 'E-Commerce Commands',
    description: 'Read the following common Twi e-commerce commands aloud clearly.',
    prompts: scriptA,
  },
  {
    id: 'ScriptBU',
    title: 'Numbers, Dates & Shopping Time',
    description: 'Read the following numbers, days, months, and time-related shopping phrases.',
    prompts: scriptB,
  },
  {
    id: 'ScriptCU',
    title: 'Online Shopping Situations',
    description: 'Read phrases related to online shopping, products, payments, and delivery.',
    prompts: scriptC,
  },
  {
    id: 'ScriptDU',
    title: 'Complex Shopping Scenarios',
    description: 'Read these more complex e-commerce sentences and questions.',
    prompts: scriptD,
  },
  {
    id: 'SpontaneousU',
    title: 'E-Commerce Spontaneous Speech',
    description: 'For the next prompts, read the topic about online shopping, then speak freely and naturally about it in Twi for 30-60 seconds. DO NOT read the instructions aloud.',
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
