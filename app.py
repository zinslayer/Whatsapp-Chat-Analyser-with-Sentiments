import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
st.sidebar.title("Whatsapp Chat Analyser")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

# Main heading
st. markdown("<h1 style='text-align: center; color: black;'>Whatsapp Chat Analysis with Sentiment </h1>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     data = bytes_data.decode("utf-8")
     df = preprocessor.preprocess(data)
     #st.text(data)
     
     #st.dataframe(df)
      # fetch unique users
     user_list = df['user'].unique().tolist()
     user_list.remove('group_notification')
     user_list.sort()
     user_list.insert(0,"Overall")

     selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

     if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages,num_words,num_media,num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Words")
            st.title(num_words)
        with col3:
             st.header('Media shared')
             st.title(num_media)
        with col4:
            st.header('Links shared')
            st.title(num_links)
    
        if selected_user == 'Overall':
            st.title('Most Busy Users with percentage')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.beta_columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        #wordcloud
        st.title("Wordcloud")
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)
    

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.beta_columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.beta_columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

         # Object
        sentiments = SentimentIntensityAnalyzer()
        # Creating different columns for (Positive/Negative/Neutral)
        df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]] # Positive
        df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]] # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]] # Neutral

        # To indentify true sentiment per row in message column
        def sentiment(d):
         if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
         if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
         if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

         # Creating new column & Applying function
        st.title("Daywise most Positive chats")    
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        daywise_positive_chats= helper.daywise_positive_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(daywise_positive_chats.index, daywise_positive_chats.values, color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)  

        # Creating new column & Applying function
        st.title("Monthwise most Positive chats")
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        monthwise_positive_chats= helper.monthwise_positive_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(monthwise_positive_chats.index, monthwise_positive_chats.values, color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daywise most Negative chats")    
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        daywise_negative_chats= helper.daywise_negative_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(daywise_negative_chats.index, daywise_negative_chats.values, color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig) 

        # Creating new column & Applying function
        st.title("Monthwise most Negative chats")
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        monthwise_negative_chats= helper.monthwise_negative_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(monthwise_negative_chats.index, monthwise_negative_chats.values, color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        st.title("Daywise most Neutral chats")    
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        daywise_neutral_chats= helper.daywise_neutral_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(daywise_neutral_chats.index, daywise_neutral_chats.values, color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig) 

        # Creating new column & Applying function
        st.title("Monthwise most Neutral chats")
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)  
        monthwise_neutral_chats= helper.monthwise_neutral_chats(selected_user, df)   
        fig, ax = plt.subplots()
        ax.bar(monthwise_neutral_chats.index, monthwise_neutral_chats.values, color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.percentage(df, 1)
                
                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.percentage(df, 0)
                
                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.percentage(df, -1)
                
                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)