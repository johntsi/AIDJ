from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

class EmoticonCloud:
    '''
    Taken from the answer of Antoine Dubuis in the following stackoverflow question
    https://stackoverflow.com/questions/66473771/wordcloud-for-only-emojis

    Generates a WordCloud for emoticons using font that support them
    '''
    def __init__(self, font_path='Symbola.otf'):
        self.font_path = font_path
        self.word_cloud = self.initialize_wordcloud()
        self.emoticon_probability = None

        
    def initialize_wordcloud(self):
        return WordCloud(font_path=self.font_path,
                               width=2000,
                               height=1000,
                               background_color='white',
                               random_state=42,
                               collocations=False)

    
    def color_func(self, word, font_size, position, orientation, random_state=None,
                   **kwargs):
        hue_saturation = '42, 88%'

        current_emoticon_probability = self.emoticon_probability[word]
        if current_emoticon_probability >= 0.10:
            opacity = 50
        else:
            opacity = 75 - current_emoticon_probability/0.2 * 5
        return f"hsl({hue_saturation},{opacity}%)"

    def generate(self, frequencies: Dict[str, float], filepath: Path) -> None:

        total_count = sum(frequencies.values())
        self.emoticon_probability = {emoticon: count/total_count for emoticon, count in frequencies.items()}

        wc = self.word_cloud.generate_from_frequencies(frequencies)
        
        plt.figure(figsize=(20,10))
        plt.imshow(wc.recolor(color_func=self.color_func, random_state=42))
        plt.axis("off")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()