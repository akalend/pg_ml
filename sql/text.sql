SELECT * FROM  
	ml_text_predict('rating_txt.cbm','rating_txt',
	'{rating_MPAA,studio,fresh,critic,top_critic,publisher}',
	'{synopsis,genre,director,writer,review}',
	'genre');

-- SELECT * FROM  
-- 	ml_text_predict_query ('rating_txt.cbm','SELECT * FROM rating_txt',
-- 	'{rating_MPAA,studio,fresh,critic,top_critic,publisher}',
-- 	'{synopsis,genre,director,writer,review}');