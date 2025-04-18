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
  { id: 'ScriptA_9', type: 'scripted', text: 'Yɛ frɛ wo sɛn?', meaning: 'What are you called?' },
  { id: 'ScriptA_10', type: 'scripted', text: 'Wo din de sɛn?', meaning: 'What is your name?' },
  { id: 'ScriptA_11', type: 'scripted', text: 'Me din de [Your First Name].', meaning: 'My name is [Your First Name].' },
  { id: 'ScriptA_12', type: 'scripted', text: 'Bra ha.', meaning: 'Come here.' },
  { id: 'ScriptA_13', type: 'scripted', text: 'Kɔ hɔ.', meaning: 'Go there.' },
  { id: 'ScriptA_14', type: 'scripted', text: 'Kɔ w’anim.', meaning: 'Go forward.' },
  { id: 'ScriptA_15', type: 'scripted', text: 'San w’akyi.', meaning: 'Go back.' },
  { id: 'ScriptA_16', type: 'scripted', text: 'Tena ase.', meaning: 'Sit down.' },
  { id: 'ScriptA_17', type: 'scripted', text: 'Gyina hɔ.', meaning: 'Stand there.' },
  { id: 'ScriptA_18', type: 'scripted', text: 'Sɔre gyina hɔ.', meaning: 'Stand up.' },
  { id: 'ScriptA_19', type: 'scripted', text: 'Fa kyɛ me.', meaning: 'Forgive me.' },
  { id: 'ScriptA_20', type: 'scripted', text: 'Fa ma me.', meaning: 'Give it to me.' },
  { id: 'ScriptA_21', type: 'scripted', text: 'Ɛyɛ.', meaning: 'It is good / Okay.' },
  { id: 'ScriptA_22', type: 'scripted', text: 'Ɛnyɛ.', meaning: 'It is not good.' },
  { id: 'ScriptA_23', type: 'scripted', text: 'Ɛnyɛ nokware', meaning: 'It is not right/true.' },
  { id: 'ScriptA_24', type: 'scripted', text: 'Boa me!', meaning: 'Help me!' },
  { id: 'ScriptA_25', type: 'scripted', text: 'Me pa wo kyɛw.', meaning: 'Please / Excuse me.' },
  { id: 'ScriptA_26', type: 'scripted', text: 'Kafra.', meaning: 'Sorry / Excuse me.' },
  { id: 'ScriptA_27', type: 'scripted', text: 'Mesrɛ wo.', meaning: 'Please / I beg you.' },
  { id: 'ScriptA_28', type: 'scripted', text: 'Mesrɛ wo, tie me.', meaning: 'Please, listen to me.' },
  { id: 'ScriptA_29', type: 'scripted', text: 'Me pa wo kyɛw fakyɛ me.', meaning: 'Please forgive me.' },
  { id: 'ScriptA_30', type: 'scripted', text: 'Me honam yɛ me ya.', meaning: 'My body hurts / I feel sick.' },
  { id: 'ScriptA_31', type: 'scripted', text: 'Frɛ dɔkota no ma me', meaning: 'Call the doctor for me.' },
  { id: 'ScriptA_32', type: 'scripted', text: 'Me werɛ afi.', meaning: 'I have forgotten.' },
  { id: 'ScriptA_33', type: 'scripted', text: 'Menhuu no.', meaning: 'I haven\'t seen him' },
  { id: 'ScriptA_34', type: 'scripted', text: 'M\'anhuu no.', meaning: 'I didn\'t see it/him/her.' },
  { id: 'ScriptA_35', type: 'scripted', text: 'Mente aseɛ.', meaning: 'I don\'t understand.' },
  { id: 'ScriptA_36', type: 'scripted', text: 'Me te aseɛ.', meaning: 'I understand.' },
  { id: 'ScriptA_37', type: 'scripted', text: 'Kasa brɛw', meaning: 'Speak slowly.' },
  { id: 'ScriptA_38', type: 'scripted', text: 'Kasa den.', meaning: 'Speak loudly / firmly.' },
  { id: 'ScriptA_39', type: 'scripted', text: 'Adɛn nti na woreyɛ saa?', meaning: 'Why are you doing that?' },
  { id: 'ScriptA_40', type: 'scripted', text: 'Ɛhe na ɔwɔ?', meaning: 'Where is he/she?' },
  { id: 'ScriptA_41', type: 'scripted', text: 'Wobɛtumi aboa me anaa?', meaning: 'Can you help me?' }
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
  { id: 'ScriptB_12', type: 'scripted', text: 'Du baako', meaning: 'Eleven' },
  { id: 'ScriptB_13', type: 'scripted', text: 'Du mmienu', meaning: 'Twelve' },
  { id: 'ScriptB_14', type: 'scripted', text: 'Du mmiɛnsa', meaning: 'Thirteen' },
  { id: 'ScriptB_15', type: 'scripted', text: 'Aduonu', meaning: 'Twenty' },
  { id: 'ScriptB_16', type: 'scripted', text: 'Aduonu baako', meaning: 'Twenty-one' },
  { id: 'ScriptB_17', type: 'scripted', text: 'Aduasa', meaning: 'Thirty' },
  { id: 'ScriptB_18', type: 'scripted', text: 'Aduanan', meaning: 'Forty' },
  { id: 'ScriptB_19', type: 'scripted', text: 'Aduonum', meaning: 'Fifty' },
  { id: 'ScriptB_20', type: 'scripted', text: 'Aduosia', meaning: 'Sixty' },
  { id: 'ScriptB_21', type: 'scripted', text: 'Aduɔson', meaning: 'Seventy' },
  { id: 'ScriptB_22', type: 'scripted', text: 'Aduowɔtwe', meaning: 'Eighty' },
  { id: 'ScriptB_23', type: 'scripted', text: 'Aduokron', meaning: 'Ninety' },
  { id: 'ScriptB_24', type: 'scripted', text: 'Ɔha', meaning: 'One hundred' },
  { id: 'ScriptB_25', type: 'scripted', text: 'Apem', meaning: 'One thousand' },
  { id: 'ScriptB_26', type: 'scripted', text: 'Ɔpepem', meaning: 'One million' },
  { id: 'ScriptB_27', type: 'scripted', text: 'Ɛyɛ dɔnhwere mmienu.', meaning: 'It is two o\'clock.' },
  { id: 'ScriptB_28', type: 'scripted', text: 'Meda nnɔnkron.', meaning: 'I sleep at nine o\'clock.' },
  { id: 'ScriptB_29', type: 'scripted', text: 'Ɛnnɛ yɛ Benada.', meaning: 'Today is Tuesday.' },
  { id: 'ScriptB_30', type: 'scripted', text: 'Ɛdwoada', meaning: 'Monday' },
  { id: 'ScriptB_31', type: 'scripted', text: 'Ɛbenada', meaning: 'Tuesday' },
  { id: 'ScriptB_32', type: 'scripted', text: 'Wukuada', meaning: 'Wednesday' },
  { id: 'ScriptB_33', type: 'scripted', text: 'Yawoada', meaning: 'Thursday' },
  { id: 'ScriptB_34', type: 'scripted', text: 'Fiada', meaning: 'Friday' },
  { id: 'ScriptB_35', type: 'scripted', text: 'Memeneda', meaning: 'Saturday' },
  { id: 'ScriptB_36', type: 'scripted', text: 'Kwasiada', meaning: 'Sunday' },
  { id: 'ScriptB_37', type: 'scripted', text: 'Ɔpɛpɔn', meaning: 'January' },
  { id: 'ScriptB_38', type: 'scripted', text: 'Ɔgyefoɔ', meaning: 'February' },
  { id: 'ScriptB_39', type: 'scripted', text: 'Ɔbɛnem', meaning: 'March' },
  { id: 'ScriptB_40', type: 'scripted', text: 'Oforisuo', meaning: 'April' },
  { id: 'ScriptB_41', type: 'scripted', text: 'Kotonimma', meaning: 'May' },
  { id: 'ScriptB_42', type: 'scripted', text: 'Ayɛwohomumɔ', meaning: 'June' },
  { id: 'ScriptB_43', type: 'scripted', text: 'Kitawonsa', meaning: 'July' },
  { id: 'ScriptB_44', type: 'scripted', text: 'Ɔsanaa', meaning: 'August' },
  { id: 'ScriptB_45', type: 'scripted', text: 'Ɛbɔ', meaning: 'September' },
  { id: 'ScriptB_46', type: 'scripted', text: 'Ahinime', meaning: 'October' },
  { id: 'ScriptB_47', type: 'scripted', text: 'Obubuo', meaning: 'November' },
  { id: 'ScriptB_48', type: 'scripted', text: 'Ɔpɛnimma', meaning: 'December' },
  { id: 'ScriptB_49', type: 'scripted', text: 'Yei yɛ sɛn?', meaning: 'How much is this?' },
  { id: 'ScriptB_50', type: 'scripted', text: 'Ɛyɛ Ghana sidi aduonum.', meaning: 'It is fifty Ghana Cedis.' },
  { id: 'ScriptB_51', type: 'scripted', text: 'Mɛtua sidi ɔha ne aduasa enum.', meaning: 'I will pay one hundred and thirty-five cedis.' },

]

const scriptC: ScriptPrompt[] = [
  { id: 'ScriptC_1', type: 'scripted', text: 'Mepaakyɛw, ntoosi no boɔ yɛ sɛn?', meaning: 'Please, how much are the tomatoes?' },
  { id: 'ScriptC_2', type: 'scripted', text: 'Mepaakyɛw, bankye no boɔ yɛ sɛn?', meaning: 'Please, how much is the cassava?' },
  { id: 'ScriptC_3', type: 'scripted', text: 'Te so kakra ma me.', meaning: 'Reduce the price a little for me.' },
  { id: 'ScriptC_4', type: 'scripted', text: 'Borɔdeɛ mmiɛnsa bɛyɛ sɛn?', meaning: 'How much will three plantains be?' },
  { id: 'ScriptC_5', type: 'scripted', text: 'Ma me kilo mmienu.', meaning: 'Give me two kilos.' },
  { id: 'ScriptC_6', type: 'scripted', text: 'Menya nkateɛ a y\'atoto no bi anaa?', meaning: 'Can I get some of the roasted groundnuts?' },
  { id: 'ScriptC_7', type: 'scripted', text: 'Merepɛ ankaa atɔ.', meaning: 'I am looking for oranges to buy.' },
  { id: 'ScriptC_8', type: 'scripted', text: 'Fa nkyene paketɛ baako ma me.', meaning: 'Give me one packet of salt.' },
  { id: 'ScriptC_9', type: 'scripted', text: 'Mepaakyɛw, fa aduane no brɛ me.', meaning: 'Please, bring me the food.' },
  { id: 'ScriptC_10', type: 'scripted', text: 'Aduane bɛn na wowɔ nnɛ?', meaning: 'What food do you have today?' },
  { id: 'ScriptC_11', type: 'scripted', text: 'Me pɛ fufuo ne nkrakra nkwan.', meaning: 'I want fufu with light soup.' },
  { id: 'ScriptC_12', type: 'scripted', text: 'Nsuo no yɛ nwunu dodo.', meaning: 'The water is too cold.' },
  { id: 'ScriptC_13', type: 'scripted', text: 'Ma me mako kakra.', meaning: 'Give me a little pepper.' },
  { id: 'ScriptC_14', type: 'scripted', text: 'Yɛbɛtumi adi aduane yi wɔ ha?', meaning: 'Can we eat this food here?' },
  { id: 'ScriptC_15', type: 'scripted', text: 'Wobɛtumi akyekyere deɛ aka no ama me?', meaning: 'Can you pack the leftovers for me?' },
  { id: 'ScriptC_16', type: 'scripted', text: 'Me ho mfa me.', meaning: 'I am not feeling well.' },
  { id: 'ScriptC_17', type: 'scripted', text: 'Meyam yɛ me ya.', meaning: 'My stomach hurts.' },
  { id: 'ScriptC_18', type: 'scripted', text: 'Me ti pae me', meaning: 'My head hurts / I have a headache.' },
  { id: 'ScriptC_19', type: 'scripted', text: 'Meyam keka me efiri nnora.', meaning: 'My stomach has been hurting since yesterday.' },
  { id: 'ScriptC_20', type: 'scripted', text: 'Ɛhe na yɛ bɛ nya aduro atɔ?', meaning: 'Where can we buy medicine?' },
  { id: 'ScriptC_21', type: 'scripted', text: 'Me hia dɔkota ntɛm ara!', meaning: 'I need a doctor immediately!' },
  { id: 'ScriptC_22', type: 'scripted', text: 'Me yam yɛ me ya', meaning: 'I have diarrhoea / stomach upset.' },
  { id: 'ScriptC_23', type: 'scripted', text: 'M’aba adwuma.', meaning: 'I have come to work.' },
  { id: 'ScriptC_24', type: 'scripted', text: 'Mereyɛ nhyiamudeɛ.', meaning: 'I am having a meeting.' },
  { id: 'ScriptC_25', type: 'scripted', text: 'Yɛwɔ nhyiamu awia nnɔn mmienu.', meaning: 'We have a meeting at 2 PM.' },
  { id: 'ScriptC_26', type: 'scripted', text: 'Mɛtumi de kɔmputa yi ayɛ m\'adwuma?', meaning: 'Can I use this computer to do my work?' },
  { id: 'ScriptC_27', type: 'scripted', text: 'Mesrɛ wo, print krataa yi ma me.', meaning: 'Please, print this paper for me.' },
  { id: 'ScriptC_28', type: 'scripted', text: 'Merekɔ akɔdi m\'awia aduane wɔ adwuma mu.', meaning: 'I am going to eat my lunch at work.' },
  { id: 'ScriptC_29', type: 'scripted', text: 'Kaa bɛn na ɛrekɔ Nkran?', meaning: 'Which car/bus goes to Accra?' },
  { id: 'ScriptC_30', type: 'scripted', text: 'Gyina wɔ beaeɛ a ɛbɛn ha ma me.', meaning: 'Stop for me at the place nearby.' },
  { id: 'ScriptC_31', type: 'scripted', text: 'M\'ani agye m\'akwantuo yi ho.', meaning: 'I enjoyed this journey.' },
  { id: 'ScriptC_32', type: 'scripted', text: 'Trotro sika no yɛ sɛn?', meaning: 'How much is the Trotro fare?' },
  { id: 'ScriptC_33', type: 'scripted', text: 'Yɛadu?', meaning: 'Have we arrived?' },
  { id: 'ScriptC_34', type: 'scripted', text: 'Yɛabɛn?', meaning: 'Are we close?' },
]
const scriptD: ScriptPrompt[] = [
  { id: 'ScriptD_1', type: 'scripted', text: 'Sɛ nsuo tɔ a, yɛntumi nkɔ afuom.', meaning: 'If it rains, we cannot go to the farm.' },
  { id: 'ScriptD_2', type: 'scripted', text: 'Mekɔɔ Kumasi nanso m\'anhu Kofi.', meaning: 'I went to Kumasi but I didn\'t see Kofi.' },
  { id: 'ScriptD_3', type: 'scripted', text: 'Ɛkɔm de me enti merepɛ biribi adi.', meaning: 'I am hungry so I am looking for something to eat.' },
  { id: 'ScriptD_4', type: 'scripted', text: 'Ɔbaa ha efiri sɛ ɔpɛ sɛ ɔhu wo.', meaning: 'He/She came here because he/she wants to see you.' },
  { id: 'ScriptD_5', type: 'scripted', text: 'Akwadaa a w\'atenase no yɛ me ba.', meaning: 'The child who sat down is my child.' },
  { id: 'ScriptD_6', type: 'scripted', text: 'Ɔkaeɛ sɛ ɔbɛba ɔkyena anɔpa.', meaning: 'He/She said that he/she will come tomorrow morning.' },
  { id: 'ScriptD_7', type: 'scripted', text: 'W’awofoɔ te ase anaa?', meaning: 'Are your parents alive?' },
  { id: 'ScriptD_8', type: 'scripted', text: 'Wo wɔ anuannom mmarima anaa mmaa sɛn?', meaning: 'How many brothers or sisters do you have?' },
  { id: 'ScriptD_9', type: 'scripted', text: 'Me nuabarima panin yɛ ɔkyerɛkyerɛni.', meaning: 'My elder brother is a teacher.' },
  { id: 'ScriptD_10', type: 'scripted', text: 'M\'adamfo pa na ɔboa me berɛ a mehia mmoa.', meaning: 'My best friend is the one who helps me when I need help.' },
  { id: 'ScriptD_11', type: 'scripted', text: 'M\'ani agye sɛ m\'ahu wo nnɛ.', meaning: 'I am happy to see you today.' },
  { id: 'ScriptD_12', type: 'scripted', text: 'Adwuma no yɛ den nanso ɛho hia.', meaning: 'The work is difficult but it is important.' },
  { id: 'ScriptD_13', type: 'scripted', text: 'Mensusu sɛ ɔbɛtumi awie nnɛ.', meaning: 'I don\'t think he/she can finish today.' },
  { id: 'ScriptD_14', type: 'scripted', text: 'Ɛwɔ sɛ yɛ boa yɛn ho yɛn ho.', meaning: 'We must help each other.' },
  { id: 'ScriptD_15', type: 'scripted', text: 'Me werɛ aho pa ara.', meaning: 'I am very sad.' },
  { id: 'ScriptD_16', type: 'scripted', text: 'M\'abrɛ pa ara.', meaning: 'I am very tired.' },
  { id: 'ScriptD_17', type: 'scripted', text: 'Ɔpanin no pɛ sɛ ɔne wo kasa.', meaning: 'The boss wants to talk to you.' },
  { id: 'ScriptD_18', type: 'scripted', text: 'Mepɛ sɛ mesua Twi kasa no yie.', meaning: 'I want to learn the Twi language well.' },
  { id: 'ScriptD_19', type: 'scripted', text: 'Kenkan nwoma yi ma me.', meaning: 'Read this book for me.' },
  { id: 'ScriptD_20', type: 'scripted', text: 'Yɛn fie no so paa na ɛwɔ abɔnten so.', meaning: 'Our house is very big and it is on the street.' },
  { id: 'ScriptD_21', type: 'scripted', text: 'Dua tenten bi si sukuu no akyi.', meaning: 'A tall tree stands behind the school.' },
  { id: 'ScriptD_22', type: 'scripted', text: 'Adɛn nti na wo kyɛeɛ?', meaning: 'Why were you late?' },
  { id: 'ScriptD_23', type: 'scripted', text: 'Ɛbɛyɛ nnɔn sɛn ansa na woawie?', meaning: 'What time will it reach before you finish? (What time will you finish?)' },
  { id: 'ScriptD_24', type: 'scripted', text: 'Wo ne hwan na ɛrekɔ?', meaning: 'With whom are you going?' },
  { id: 'ScriptD_25', type: 'scripted', text: 'Mpɛn ahe na wodidi?', meaning: 'How many times do you eat in a day?' },
  { id: 'ScriptD_26', type: 'scripted', text: 'Nnora, menantee kɔɔ kurom.', meaning: 'Yesterday, I walked to town.' },
  { id: 'ScriptD_27', type: 'scripted', text: 'Ɔkyena yɛbɛdi agorɔ wɔ agoprama so.', meaning: 'Tomorrow we will play on the field.' },
  { id: 'ScriptD_28', type: 'scripted', text: 'Berɛ a na meredidi no, me nua no baeɛ.', meaning: 'While I was eating, my sibling came.' },
  { id: 'ScriptD_29', type: 'scripted', text: 'Ɔtumi ka Borɔfo ne Twi fɛfɛɛfɛ.', meaning: 'He/She can speak English and Twi beautifully/fluently.' },
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
