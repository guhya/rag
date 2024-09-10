import json

class FilmScore:
    def __init__(self, film_id: int, score: float):
        self.film_id = film_id
        self.score = score

        self.title = None
        self.description = None
        self.llm_summary = None

    def __hash__(self):
        # Use film_id as the basis for hashing
        return hash(self.film_id)

    def __eq__(self, other):
        # Two FilmScore instances are equal if they have the same film_id
        if isinstance(other, FilmScore):
            return self.film_id == other.film_id
        return False

    def __repr__(self):
        return (f"FilmScore(film_id={self.film_id}, score={self.score}, "
                f"title={self.title}, description={self.description}, llm_summary={self.llm_summary})")
    
    def to_dict(self):
        return {
            "film_id": self.film_id
            , "score": self.score
            , "title": self.title
            , "description": self.description
            , "llm_summary": self.llm_summary
        }
    
    def to_str(self):
        return f"film_id={self.film_id}, score={self.score}, title={self.title}, description={self.description}"
