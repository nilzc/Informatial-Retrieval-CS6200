# Objectives
1. Write a web crawler to crawl around 40,000 pages, which are related to a particular topic.
    * Comform strictly to the politeness policy
2. Work with a team, each team member will be given several seed URLs as the start for his/her crawler.
    * After all crawlers finish, merge each member's pages together.
3. Design a vertical search engine using Elasticsearch (a graphic user interface).

# Crawling Documents
1. Set up Elasticsearch to do indexing.
2. Frontier management:
    * Start from the seed URLs, and crawl at least 40,000 pages.
    * Use a modified BFS method to find new pages
        * Best-first, always crawl relevant pages first, use BFS "wave number" as the baseline score, add keyword count and in-links count as supplementary scores.
3. Politeness policy:
    * Protect the website you crawl and also prevent your crawler from being blocked.
    * Make no more than one HTTP request per second from any given domain. But it's okay to crawl multiple pages from different domains at the same time.
    * You can make a GET request for the same URL right after a HEAD request without waiting.
    * Before you crawl the first page from a given domain, fetch its robots.txt file and make sure your crawler obeys the file (permission, delay, etc.)
4. Crawl only HTML and English documents, you need to devise a way to ensure this.
5. Record out-links and in-link of each page you crawled and canonicalize them.
    * Canonicalization includes but is not limited to:
        * Convert the scheme and host to lower case.
        * Remove port 80 from http URLs, and port 443 from HTTPS URLs.
        * Make relative URLs absolute.
        * Remove the fragment.
        * Remove duplicate slashes.
6. Document processing:
    * Use a third party library to parse a HTML page (e.g. BeautifulSoup in Python)
    * Extract all ```<a>``` tags, and record them as out-links.
    * Extract all text content (you may select all ```<p>``` tags).
    * Canonicalized URL will be the DOCNO of each document.

# Merging team indexes
1. Use Elasticsearch to merge index by writting scripts when uploading docs to the server.
2. Every one should update the index independentily while the ES server is connected.
3. Use an ES cloud server or a local server with virtual area network.

# Vertical Search
Design an user interface to enter a query and fetch relevant docs from Elasticsearch server.
* Imagine you're using Google, the search result may contain page title, page URL, a preview of the page content, etc.
* You may use a local ES server or a cloud server.
