from sentiment_analysis_spanish import sentiment_analysis

sentiment = sentiment_analysis.SentimentAnalysisSpanish()
print(sentiment.sentiment("me gusta la tómbola es genial"))
print(sentiment.sentiment("paraguay no ejecutó correctamente el plan COVID"))
print(sentiment.sentiment("el club guarani nunca saldra campeón"))
print(sentiment.sentiment("el club guarani no será eliminado"))
