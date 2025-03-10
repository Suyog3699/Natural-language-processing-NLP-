# Step 1: Install necessary packages (run once)
install.packages('tm', dependencies = TRUE)
install.packages("SnowballC")
install.packages("Matrix")
install.packages("lsa")

# Step 2: Load Libraries
library(tm)
library(SnowballC)
library(Matrix)
library(lsa)

# Step 3: Define Movie Tags (Fix the list structure)
movie_tags <- c("Sci-Fi Thriller Dream Heist Leo Hardy Nolan Zimmer",  # Inception
                "Sci-Fi Drama Space Future McConaughey Nolan Zimmer",  # Interstellar
                "Action Crime Gotham Joker Bale Ledger Nolan Zimmer")  # The Dark Knight

# Step 4: Create a Text Corpus
docs <- VCorpus(VectorSource(movie_tags))

# Step 5: Preprocessing - Convert to Lowercase
docs <- tm_map(docs, content_transformer(tolower))
# Remove Punctuation
docs <- tm_map(docs, removePunctuation)
# Remove Numbers
docs <- tm_map(docs, removeNumbers)
# Remove Stopwords (Common Words)
docs <- tm_map(docs, removeWords, stopwords("en"))
# Apply Stemming (Convert words to root form)
docs <- tm_map(docs, stemDocument)

# Step 6: Create Document-Term Matrix (BoW Representation)
dtm <- DocumentTermMatrix(docs)
inspect(dtm)  # View the matrix

# Step 7: Convert DTM into a Sparse Matrix
sparse_dtm <- as.matrix(dtm)  # Convert to matrix for cosine similarity
print(sparse_dtm)  # View the matrix

# Step 8: Compute Cosine Similarity
similarity_matrix <- cosine(t(sparse_dtm))  # Transpose for correct cosine calculation
print(similarity_matrix)  # Display similarity scores

#Diagonal values are always 1 → A movie is 100% similar to itself.
#Inception vs. Interstellar = 0.428 → These two movies have 42.8% similarity based on their tags.
#Inception vs. The Dark Knight = 0.333 → Lower similarity (33.3%).
#Interstellar vs. The Dark Knight = 0.166 → The lowest similarity (16.6%), meaning they share fewer common words.