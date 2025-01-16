DEEP DIVE DDP


2203 performs a training step
3137 contex manager che inizia il calcolo
3161 è dove calcola l'output del modello
	qui vediamo come vengono prodotti i logits:
		outputs.logits
		len(outputs.logits[0]) lunghezza di una risposta
		len(outputs.logits[0][0]) numero di token possibili in output
		test = torch.argmax(outputs.logits[0], dim = 1) #token prodotti dopo aver applicato argmax
		self.tokenizer.decode(test) #testo prodotto
3147 perform backwards




to check the memory
[i/1024**3 for i in torch.cuda.mem_get_info()]


TL:DR 

pare che avevo messo un parametro maledetto 

group_by_length: False 
e
padding_size: "right"

che stavano dando in input al modello una quantità immonda di token in input e riempiva la memoria di una gpu

unica cosa che non mi spiego, non capisco perchè capitava quasi sempre alla GPU di rank più basso 


in più se mettiamo un tokenizer troppo ampio, sfonda la memoria della gpu. 
dobbiamo regolare bene 3 cose:
- numero di esempi in ogni batch,
- gradient accumulation step
- max tokens per il tokenizzatore

inoltre va assolutamente verificato con quale grado di precisione avviene il calcolo dei gradienti ecc...


resume_from_checkpoint

per il gradient accumulation steps --> https://medium.com/@harshit158/gradient-accumulation-307de7599e87
				       https://discuss.pytorch.org/t/vram-usage-increase-with-more-gradient-accumulation-steps/180729
per il warm up ratio --> https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps


migliorare:
- numero di epoch
- moduli su cui montare 
- precisione



