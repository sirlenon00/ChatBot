# -*- coding: utf-8 -*-
# Trabalho: ChatBot para Eventos em Geral, com o objetivo tirar as dúvidas dos clientes.
# Programadores: Matheus de Almeida Souza e Sirlenon de Araujo Macedo

from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,strip_accents='unicode')

# corpus recebe as sentenças
corpus = ['Quem vai participar do evento?','Quem vai participar do evento esse ano?','Quais serão os convidados esse ano?','Quem vai vir no evento?','Quem vai estar no evento?','Quem vai vir esse ano?', 'Quem são os participantes?',
          'Qual será o cantor esse ano?','Qual será o cantor?', 'Quem seram os cantores?','qual o nome dos cantores?','Quais seram os cantores?',
          'O que vai ter no evento?','O que é o evento?','Quais serão as atrações desse ano?','O evento é de quê?','Saber desse tal evento?','Pode me falar sobre o que é o evento?','saber mais sobre o evento?','Voce poderia me informar melhor a respeito desse evento?','Quais são as atrações do evento?',
          'Olá','Oi','Oii','Oiii','Bom dia','Boa noite','Boa tarde','Eae',
          'Estou com uma dúvida','Pode me esclarecer uma dúvida?',
          'Pode me ajudar?','Por favor, pode me ajudar?',
          'Quanto é o ingresso?','Quanto custa o ingresso?','Qual o valor do ingresso?','Quanto que ta o valor do lote?','Quanto ta o ingresso individual?','Quanto ta meia entrada?','Qual o preço do Lote?', 'respeito dos ingressos','Saber o preço dos ingressos',
          'Quem é você?','Estou falando com quem?','O que é você?','Qual é o seu nome?','Quem é tu?','Tu é quem?',
          'Quem é Alok?','O que é Alok?','Quem é esse tal de alok?', 'Alok é cantor?', 'Alok é dj?',
          'Quem é Rihanna?','O que é Rihanna?','Quem é Riana?','Quem é esse tal de Rihanna?','Rihanna é cantora?',
          'Quem é LinkPark?','O que é LinkPark?','Quem é esse tal de LinkPark?','LinkinPark é uma banda?',
          'Quais os horários', 'qual os horários do show','que horas vai ser o circu di solei?','Que horas vai comecar o circu di solei?','que horas termina o circu di solei?','acaba que horas o circo di solei?','Qual a programação do Cirque du Soleil','Que horas é o Cirque du Soleil','Que horas começa o Show?','Que horas começa o Evento?','Que horas começa?','Quando começa?','Que dia vai ser?','Que dia começa o evento?', 'que horas é o show do alok?','que horas vai tocar alok?','que horas é o show da rihanna?','que horas vai ter linkin park?','que horas vai ser o show do linkin park?','que horas vai ser o show do alok?','que horas vai ser o show da rihanna?','que horas é o alok?','quando vai ser o alok?','quando vai ser a rihanna?','quando vai ser o linkin park?', 'Qual será a data do evento?', 'Quando será o show?', 'Qual será as datas dos shows?','Quais seram os dias do evento?','Qual é o horário do show?',
          'Qual é o local do show?','Qual é o local do evento?','Onde que vai ser o evento?','Onde que vai ser o show?', 'Onde será realizado esse evento?',
          'Onde compra o ingresso?','Como faz para comprar o ingresso?','Onde que vende o ingresso?','Você vende ingresso?','vc vende ingresso?',
          'Muito Obrigado!','Valeu','Obrigado pela ajuda!','Vlw','Obg',
          'Perdi o ingresso','Esqueci o Ingresso', 'Eu acabei perdendo meu ingresso.', 'O que eu faço quando eu perco meu ingresso',
          'O que ta incluso no ingresso vip?','área vip tem o que?','Área vip tem direito a que?','Como é esse ingresso vip?',
          'Quero o reembolso do ingresso','como faço para ter o reembolso do ingresso?','em que situações tenho direito a reembolso?','se o cantor faltar, tenho direito a estorno do valor?','quero estorno do meu dinheiro',
          'O local tem acessibilidade?','Tem vaga para deficientes?','O evento tem acessibilidade para deficientes?','O evento tem acessibilidade para grávidas?','O evento tem acessibilidade para gestantes?','O evento tem acessibilidade para idosos?', 'No evento irá ter acessibilidade?',
          'Como conseguir a segunda via do ingresso?', 'Como eu consigo a 2 via do ingresso?', 'Como faço para pedir a minha 2 via do ingresso?',
          'Como posso entrar em contato com o local?', 'Qual é o telefone do local?','Tem telefone?','Tem site?','Qual é o site do local?', 'O local tem celular para contato?',
          'O evento aceita meia-entrada?','Vocês aceitam meia entrada?','aceita meia entrada?','Estudante tem direito a meia-entrada?',
          'O que acontece se o show é cancelado?','Se o evento cancelar?','Se o show cancelar?',
          'xau','Tchau','Até mais','Adeus','Bye','Good Bye','Flw','Até logo','chau',
          'Quais são as regras do local?','Pode levar animal?','pode levar animais?','Pode levar cachorro?','pode entrar sem camisa?','Pode ir sem camisa?','Pode entrar de mochila?','Pode levar garrafa?','Pode levar canivete?','Ir de arma?','Pode entrar de capacete?','Quais são as leis do show?','O que não posso levar no show',
          'Quem é circo di solei?', 'o que é circu di solei?','o que é circo de solei?','o que circus de solei?','Quem é Cirque du Soleil','o que é Cirque du Soleil?',
          'O local terá segurança?','Terá saída de emergência?','Terá extintor de incêndio no show?','O show ira ter policiais?','O local terá vigilantes?',
          'Vai ter banheiro?','O local irá fornecer banheiro?','Terá banheiro feminino?','Terá banheiro masculino?','Ira ter banheiro para deficientes?','O local tem banheiro?',
          'Terá comida no show?','Vai ter comida no evento?','Irá ter bebida no show?','Irá ter cerveja no evento?','Vai ter salgados no evento?','Terá água no evento?','Vai ter agua no Show?',
          'O show vai ser openbar?','É OpenBar?','Irá ser openBar?','Será open bar?',
          'Qual será as formas de pagamento?','Aceita cartão de debito ou crédito?','Aceita dinheiro?','Posso pagar com o cartão','O show aceitará cartao de crédito?','Como posso efetuar o pagamento?',
          'Qual será o valor da comida?','Qual será o preço do salgado?','Qual vai ser o preco da pastel?','Qual será o preço do lanche?','Quais são os preços das comidas?',
          'Vai ter estacionamento no show?','Terá estacionamento no show?', 'O local ira ter estacionamento', 'O show fornecerá estacionamento',
          'quais bebidas?','Qual será o valor da bebida?','Qual será o preco da cerveja?','Qual vai ser o preço da água?','Qual será o preço do energético?','Quais são os preços das bebidas?'
          ]
x = vectorizer.fit_transform(corpus)

import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

# y recebe as intenções
y = np.array(['SobreParticipantes','SobreParticipantes','SobreParticipantes','SobreParticipantes','SobreParticipantes','SobreParticipantes','SobreParticipantes',
              'SobreCantor','SobreCantor', 'SobreCantor','SobreCantor','SobreCantor',
              'SobreEvento','SobreEvento','SobreEvento','SobreEvento','SobreEvento','SobreEvento','SobreEvento','SobreEvento','SobreEvento',
              'Saudacao','Saudacao','Saudacao','Saudacao','Saudacao','Saudacao','Saudacao','Saudacao',
              'Duvida','Duvida',
              'Ajuda','Ajuda',
              'PrecoIngresso','PrecoIngresso','PrecoIngresso','PrecoIngresso','PrecoIngresso','PrecoIngresso', 'PrecoIngresso','PrecoIngresso','PrecoIngresso',
              'QuemSouEu','QuemSouEu','QuemSouEu','QuemSouEu','QuemSouEu','QuemSouEu',
              'QuemEAlok','QuemEAlok','QuemEAlok','QuemEAlok','QuemEAlok',
              'QuemERihanna','QuemERihanna','QuemERihanna','QuemERihanna','QuemERihanna',
              'QuemELinkPark','QuemELinkPark','QuemELinkPark','QuemELinkPark',
              'DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento','DataEvento',
              'LocalEvento','LocalEvento','LocalEvento','LocalEvento','LocalEvento',
              'ComprarIngresso','ComprarIngresso','ComprarIngresso','ComprarIngresso','ComprarIngresso',
              'Agradecimento','Agradecimento','Agradecimento','Agradecimento','Agradecimento',
              'PerdeuIngresso','PerdeuIngresso','PerdeuIngresso','PerdeuIngresso',
              'AcessoVip','AcessoVip','AcessoVip','AcessoVip',
              'Reembolso','Reembolso','Reembolso','Reembolso','Reembolso',
              'Acessibilidade','Acessibilidade','Acessibilidade','Acessibilidade','Acessibilidade','Acessibilidade','Acessibilidade',
              'SegundaVia','SegundaVia','SegundaVia',
              'Contato','Contato','Contato','Contato','Contato','Contato',
              'meiaEntrada','meiaEntrada','meiaEntrada','meiaEntrada',
              'Cancelado','Cancelado','Cancelado',
              'Despedida','Despedida','Despedida','Despedida','Despedida','Despedida','Despedida','Despedida','Despedida',
              'Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras','Regras',
              'QuemECirco','QuemECirco','QuemECirco','QuemECirco','QuemECirco','QuemECirco',
              'Seguranca','Seguranca','Seguranca','Seguranca','Seguranca',
              'Banheiro','Banheiro','Banheiro','Banheiro','Banheiro','Banheiro',
              'Comida','Comida','Comida','Comida','Comida','Comida','Comida',
              'openBar','openBar','openBar','openBar',
              'FormaPagamento','FormaPagamento','FormaPagamento','FormaPagamento','FormaPagamento','FormaPagamento',
              'precoComida','precoComida','precoComida','precoComida','precoComida',
              'Estacionamento','Estacionamento','Estacionamento','Estacionamento',
              'precoBebida','precoBebida','precoBebida','precoBebida','precoBebida', 'precoBebida'
              ])

# Validador cruzado Leave-One-Out
loo = LeaveOneOut()

# Retorna o número de iterações de divisão no validador cruzado
loo.get_n_splits(x,y)

predito = []      # Array para guardar as predições
acerto = 0        # Variável contadora de predições acertadas
erro = 0          # Variável contadora de predições erradas

# Descomente abaixo para utilizar a Classificação de Regressão Logistica
#model = LogisticRegression(solver='lbfgs', multi_class='auto')
#print("Regressão Logistica")

# Descomente abaixo para utilizar a Classificação de Arvore de Decisão
#model = DecisionTreeClassifier(random_state = 9)
#print("Arvore de Decisão")

# Descomente abaixo para utilizar a Classificação do KNN
model = KNeighborsClassifier(n_neighbors = 1)
#print("KNN")

# Neste 'for' é feito o treinamento utilizando o LeaveOneOut, para depois calcularmos o accuracy, precision e recall 
for train_index, test_index in loo.split(x):
  #print("TRAIN:", train_index, "TEST:", test_index)
  x_train, x_test = x[train_index], x[test_index]
  y_train, y_test = y[train_index], y[test_index]
  #print(x_train, x_test, y_train, y_test)
  model.fit(x_train,y_train)                # Treinamento
  predito.append(model.predict(x_test))     # Adiciona a predição do 'X_test' no vetor 'predito'
  # Atualiza numero de predicões acertadas e erradas
  if model.predict(x_test) == y_test:       
    acerto += 1
  else:
    erro += 1

# Imprime o número de acertos e erros  
#print("Acertos: ", acerto)
#print("Erros: ", erro)

#Calculo de Escore de classificação de precisão
from sklearn.metrics import accuracy_score
y_pred = predito
y_true = y
#print ("accuracy:", accuracy_score(y_true, y_pred))

#Calculo da Precisão
from sklearn.metrics import precision_score
y_pred = predito
y_true = y
#print ("precision:", precision_score(y_true, y_pred, average='micro'))

#Calculo do Recall
from sklearn.metrics import recall_score
y_pred = predito
y_true = y
#print ("recall:", recall_score(y_true, y_pred, average='micro'))

# Aqui começa a programação do chatbot, de que forma ele vai reagir as sentenças preditas
print("Pode começar a falar com o ChatBot do telegram.\n")
print("Para finalizar ou interroper o programa basta apertar as teclas ctrl + c.\n")

#importação da biblioteca telepot "pip install telepot"
import telepot

# variavel que recebe a identificação do bot do telegram
bot = telepot.Bot("1059129739:AAGM5MVTfsVXaQjdU2QpmTaXjsLgsg1-lHo")

# Função que recebe a mensagem do bot e o envia a resposta
def receber(msg):
  text = msg['text'] #variavel que recebe o texto que a pessoa manda
  _id = msg['from']['id'] #variavel que recebe a identificação da pessoa que enviou a mensagem
  print(text)
  #sair = "false"
  #while(sair == "false"):
  #usuario = input("")
  #inst = vectorizer.transform([usuario])
  inst = vectorizer.transform([text])
  intencao = model.predict(inst)
  #print(model.predict(inst))
  # Abaixo são os if e elif para respostas do bot de acordo com as intesões do usuarios
  if(text == '/start'):
    print("Bem vindo ao chatBot de tiraDuvidas!")
    bot.sendMessage(_id, "Bem vindo ao chatBot de tiraDuvidas de evento!") #função sendMessage() manda a mensagem do bot para o identificador da pessoa
  elif(intencao == ['SobreParticipantes']):    #1
    print("Os participantes desse ano serão: Cirque du Soleil, Alok, Rihanna e Linkin Park.")
    bot.sendMessage(_id, "Os participantes desse ano serão: Cirque du Soleil, Alok, Rihanna e Linkin Park.")
  elif(intencao == ['SobreCantor']):           #2
    print("Os cantores desse ano serão: Alok, Rihanna e LinkinPark.")
    bot.sendMessage(_id, "Os cantores desse ano serão: Alok, Rihanna e LinkinPark.")
  elif(intencao == ['SobreEvento']):           #3
    print("Neste ano o evento será totalmente artístico e contará com diversos cantores famosos. Também haverá atração do Circo de Soleil.")
    bot.sendMessage(_id, "Neste ano o evento será totalmente artístico e contará com diversos cantores famosos. Também haverá atração do Circo de Soleil")
  elif(intencao == ['Saudacao']):              #4
    print("Olá, em que posso ajudar?")
    bot.sendMessage(_id, "Olá, em que posso ajudar?")
  elif(intencao == ['Duvida']):                #5
    print("Está com dúvida? é só me perguntar.")
    bot.sendMessage(_id, "Está com dúvida? é só me perguntar")
  elif(intencao == ['Ajuda']):                 #6
    print("Estou aqui para te ajudar, é só perguntar.")
    bot.sendMessage(_id, "Estou aqui para te ajudar, é só perguntar.")
  elif(intencao == ['PrecoIngresso']):         #7
    print("Preço ingresso Evento 2020.\nINTEIRA\n1º - lote: R$ 120,00\n2º - lote: R$ 150,00\n3º - lote: R$ 200,00\nMEIA\n1º - lote: R$ 60,00\n2º - lote: R$ 75,00\n3º - lote: R$ 100,00\n*MEIA: Estudantes (apresentar carteira de estudante), Idosos(mais de 65 anos de idade)\nISENTO: Crianças de até 3 anos de idade.")
    bot.sendMessage(_id, "Preço ingresso Evento 2020.\nINTEIRA\n1º - lote: R$ 120,00\n2º - lote: R$ 150,00\n3º - lote: R$ 200,00\nMEIA\n1º - lote: R$ 60,00\n2º - lote: R$ 75,00\n3º - lote: R$ 100,00\n*MEIA: Estudantes (apresentar carteira de estudante), Idosos(mais de 65 anos de idade)\nISENTO: Crianças de até 3 anos de idade.")
  elif(intencao == ['QuemSouEu']):             #8
    print("Eu sou um chatbot, um atendente virtual programado para responder todas suas dúvidas referente ao evento. É um prazer servir você!")
    bot.sendMessage(_id, "Eu sou um chatbot, um atendente virtual programado para responder todas suas dúvidas referente ao evento. É um prazer servir você!")
  elif(intencao == ['QuemEAlok']):             #9
    print("Alok Achkar Peres Petrillo é um DJ e produtor musical brasileiro de música eletrônica, classificado como o 11º melhor DJ do mundo pela revista DJ Mag.")
    bot.sendMessage(_id, "Alok Achkar Peres Petrillo é um DJ e produtor musical brasileiro de música eletrônica, classificado como o 11º melhor DJ do mundo pela revista DJ Mag.")
  elif(intencao == ['QuemERihanna']):          #10
    print("Robyn Rihanna Fenty é uma cantora, compositora, atriz, empresária, filantropo e diplomata barbadense. Ela assinou contrato com a editora Def Jam Recordings após uma audição, que despertou o interesse do produtor Evan Rogers e do vice-presidente na altura da editora, Jay-Z, para a jovem artista.")
    bot.sendMessage(_id, "Robyn Rihanna Fenty é uma cantora, compositora, atriz, empresária, filantropo e diplomata barbadense. Ela assinou contrato com a editora Def Jam Recordings após uma audição, que despertou o interesse do produtor Evan Rogers e do vice-presidente na altura da editora, Jay-Z, para a jovem artista.")
  elif(intencao == ['QuemELinkPark']):         #11
    print("Linkin Park é uma banda de rock dos Estados Unidos formada em Agoura Hills, Califórnia. A formação atual da banda inclui o vocalista e multi-instrumentista Mike Shinoda, o guitarrista Brad Delson, o baixista Dave Farrell, o DJ Joe Hahn e o baterista Rob Bourdon, todos membros fundadores.")
    bot.sendMessage(_id, "Linkin Park é uma banda de rock dos Estados Unidos formada em Agoura Hills, Califórnia. A formação atual da banda inclui o vocalista e multi-instrumentista Mike Shinoda, o guitarrista Brad Delson, o baixista Dave Farrell, o DJ Joe Hahn e o baterista Rob Bourdon, todos membros fundadores.")
  elif(intencao == ['DataEvento']):            #12
    print("Será realizado entre os dias 20 á 22 de outubro. Com a seguintes datas e horários:\nDia 20\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Alok começando às 21:00h e terminando às 23:45h\nDia 21\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Rihanna começando às 21:00h e terminando às 23:45h\nDia 22\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Linkin Park começando às 21:00h e terminando às 23:45h")
    bot.sendMessage(_id, "Será realizado entre os dias 20 á 22 de outubro. Com a seguintes datas e horários:\nDia 20\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Alok começando às 21:00h e terminando às 23:45h\nDia 21\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Rihanna começando às 21:00h e terminando às 23:45h\nDia 22\n\tCirque du Soleil começando às 19:00h e terminando às 20:00h\n\tShow do Linkin Park começando às 21:00h e terminando às 23:45h")
  elif(intencao == ['LocalEvento']):           #13
    print("Endereço: Av. Pres. Castelo Branco, Portão 3 - Maracanã, Rio de Janeiro - RJ, 20271-130")
    bot.sendMessage(_id, "Endereço: Av. Pres. Castelo Branco, Portão 3 - Maracanã, Rio de Janeiro - RJ, 20271-130")
  elif(intencao == ['ComprarIngresso']):       #14
    print("Venda de ingresso somente pelo nosso site: https://www.circusmusic.com.br e no local do evento.")
    bot.sendMessage(_id, "Venda de ingresso somente pelo nosso site: https://www.circusmusic.com.br e no local do evento.")
  elif(intencao == ['Agradecimento']):         #15
    print("De nada! Eu que agradeço.")
    bot.sendMessage(_id, "De nada! Eu que agradeço.")
  elif(intencao == ['PerdeuIngresso']):        #16
    print("Em caso de perda de ingresso, é possível imprimir a segunda via no nosso site (https://www.circusmusic.com.br), ou apresentar seu RG ou CPF na entrada do evento.")
    bot.sendMessage(_id, "Em caso de perda de ingresso, é possível imprimir a segunda via no nosso site (https://www.circusmusic.com.br), ou apresentar seu RG ou CPF na entrada do evento.")
  elif(intencao == ['Estacionamento']):        #17
    print("O estacionamento é apenas destinado há carros e motos, com o valor de R$ 10,00 para ambos.\nNosso estacionamento conta com acessibilidade à idosos, gestantes e portadores de necessidades especiais")
    bot.sendMessage(_id, "O estacionamento é apenas destinado há carros e motos, com o valor de R$ 10,00 para ambos.\nNosso estacionamento conta com acessibilidade à idosos, gestantes e portadores de necessidades especiais")
  elif(intencao == ['AcessoVip']):             #18
    print("Nesse ano não teremos acesso Vip ou qualquer área privelegiada.")
    bot.sendMessage(_id, "Nesse ano não teremos acesso Vip ou qualquer área privelegiada.")
  elif(intencao == ['Reembolso']):             #19
    print("Reembolso somente presencial na vendas de ingressos do evento.")
    bot.sendMessage(_id, "Reembolso somente presencial na venda de ingressos do evento.")
  elif(intencao == ['Acessibilidade']):        #20
    print("Nosso evento conta com suporte a gestantes, idosos e portadores de necessidades especiais. Estacionamentos, banheiros, arquibancada e equipe médica")
    bot.sendMessage(_id, "Nosso evento conta com suporte a gestantes, idosos e portadores de necessidades especiais. Estacionamentos, banheiros, arquibancada e equipe médica")
  elif(intencao == ['FormaPagamento']):        #21
    print("Formas de Pagamento: Cartão de crédito (Visa e MasterCard), Cartão de débito (Elo, Visa e MasterCard) e dinheiro (somente no local do evento).")
    bot.sendMessage(_id, "Formas de Pagamento: Cartão de crédito (Visa e MasterCard), Cartão de débito (Elo, Visa e MasterCard) e dinheiro (somente no local do evento).")
  elif(intencao == ['Despedida']):             #22
    print("Até mais, qualquer coisa é só me chamar.")
    bot.sendMessage(_id, "Até mais, qualquer coisa é só me chamar.")
    #sair = "True" 
  elif(intencao == ['Contato']):               #23
    print("Você pode entrar em contato pelo:\nCelular:(21)90000-0000\nTelefone Fixo(21)3000-0000\nSite:https://www.circusmusic.com.br/contacts")
    bot.sendMessage(_id, "Você pode entrar em contato pelo:\nCelular:(21)90000-0000\nTelefone Fixo(21)3000-0000\nSite:https://www.circusmusic.com.br/contacts")
  elif(intencao == ['meiaEntrada']):           #24
    print("O direito à Meia-Entrada é garantido por leis federais e regionais. As leis federais têm abrangência em todo território nacional e as leis regionais têm eficácia restrita ao território onde foram publicadas, portanto o evento aceitará meia entrada.")
    bot.sendMessage(_id, "O direito à Meia-Entrada é garantido por leis federais e regionais. As leis federais têm abrangência em todo território nacional e as leis regionais têm eficácia restrita ao território onde foram publicadas, portanto o evento aceitará meia entrada.")
  elif(intencao == ['Cancelado']):             #25
    print("No caso de cancelamento por algum motivo, irá ter o reembolso do dinheiro gasto no ingresso.")
    bot.sendMessage(_id, "No caso de cancelamento por algum motivo, irá ter o reembolso do dinheiro gasto no ingresso.")
  elif(intencao == ['Regras']):                #26
    print("Regras:\nNão é pemitido a entrada de nenhum tipo de arma.\nNão é permitido a entrada de animais\nNão é permitido o consumo de drogas ilícitas\nObjetos proíbidos: faca, garfo, capacete, bombinhas")
    bot.sendMessage(_id, "Regras:\nNão é pemitido a entrada de nenhum tipo de arma.\nNão é permitido a entrada de animais\nNão é permitido o consumo de drogas ilícitas\nObjetos proíbidos: faca, garfo, capacete, bombinhas")
  elif(intencao == ['QuemECirco']):            #27
    print("Cirque du Soleil é uma companhia multinacional canadense de entretenimento, sediada na cidade de Montreal. Foi fundada em junho de 1984 na cidade de Baie-Saint-Paul pelos artistas de rua Guy Laliberté e Gilles Ste-Croix, sendo atualmente a maior companhia circense do mundo.")
    bot.sendMessage(_id, "Cirque du Soleil é uma companhia multinacional canadense de entretenimento, sediada na cidade de Montreal. Foi fundada em junho de 1984 na cidade de Baie-Saint-Paul pelos artistas de rua Guy Laliberté e Gilles Ste-Croix, sendo atualmente a maior companhia circense do mundo.")
  elif(intencao == ['Seguranca']):             #28
    print("No local haverá uma equipe de segurança contrato pela organização do evento e terá saídas de emergências e extintor de incêndios no local.")
    bot.sendMessage(_id, "No local haverá uma equipe de segurança contrato pela organização do evento e terá saídas de emergências e extintor de incêndios no local.")    
  elif(intencao == ['Banheiro']):              #29
    print("No local terá banheiros femininos, masculinos e para pessoas com necessidades especiais")
    bot.sendMessage(_id, "No local terá banheiros femininos, masculinos e para pessoas com necessidades especiais")
  elif(intencao == ['Comida']):                #30
    print("No evento irá ter Comida e Bedidas.\nComidas:\nSalgados\nPasteis\nLanche\nBebidas:\nCerveja\nÁgua\nEnergético.")
    bot.sendMessage(_id, "No evento irá ter Comida e Bedidas.\nComidas:\nSalgados\nPasteis\nLanche\nBebidas:\nCerveja\nÁgua\nEnergético.")
  elif(intencao == ['openBar']):               #31 
    print("O evento não será openBar, mas será vendido no local bebidas e comidas.")
    bot.sendMessage(_id, "O evento não será openBar, mas será vendido no local bebidas e comidas.")
  elif(intencao == ['precoComida']):          #32
    print("Nossas comidas:\nPastel: 5,00 Reais\nSalgados: 5,00 Reais\nLanche: 15,00 Reais.")
    bot.sendMessage(_id, "Comidas:\nPastel: 5,00 Reais\nSalgados: 5,00 Reais\nLanche: 15,00 Reais.")
  elif(intencao == ['precoBebida']):          #33
    print("Nossas bebidas serão:\nCerveja: 10,00 Reais\nÁgua: 7,00 Reais\nEnergético: 15,00 Reais.")
    bot.sendMessage(_id, "Bebidas:\nCerveja: 10,00 Reais\nÁgua: 7,00 Reais\nEnergético: 15,00 Reais.")

bot.message_loop(receber) #messageLoop() faz chamadas para getUpdates() continuamente e aplica a função de tratamento de mensagens a todas as mensagens recebidas

while True:
  pass  # esperar interrupção via teclado (Ctrl+C)  