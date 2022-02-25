package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// album represents data about a record album.
type album struct {
	ID    string `json:"id"`
	LABEL string `json:"label"`
	DATE  string `json:"date"`
	LEVEL string `json:"level"`
	DEP   string `json:"dep"`
}

// albums slice to seed record album data.
var albums = []album{
	{ID: "1", LABEL: "?", DATE: "Blue Train", LEVEL: "John Coltrane", DEP: "sad"},
	{ID: "2", LABEL: "?", DATE: "Jeru", DEP: "dad", LEVEL: "Gerry Mulligan"},
	{ID: "3", LABEL: "?", DATE: "Sarah Vaughan and Clifford Brown", LEVEL: "Sarah Vaughan", DEP: "dsad"},
}

func main() {
	router := gin.Default()
	router.GET("/albums", getAlbums)
	router.GET("/albums/:id", getAlbumByID)
	router.POST("/albums", postAlbums)

	router.Run("localhost:8080")
}

// getAlbums responds with the list of all albums as JSON.
func getAlbums(c *gin.Context) {
	c.IndentedJSON(http.StatusOK, albums)
}

// postAlbums adds an album from JSON received in the request body.
func postAlbums(c *gin.Context) {
	var newAlbum album

	// Call BindJSON to bind the received JSON to
	// newAlbum.
	if err := c.BindJSON(&newAlbum); err != nil {
		return
	}

	// Add the new album to the slice.
	albums = append(albums, newAlbum)
	c.IndentedJSON(http.StatusCreated, newAlbum)
}

// getAlbumByID locates the album whose ID value matches the id
// parameter sent by the client, then returns that album as a response.
func getAlbumByID(c *gin.Context) {
	id := c.Param("id")

	// Loop through the list of albums, looking for
	// an album whose ID value matches the parameter.
	for _, a := range albums {
		if a.ID == id {
			c.IndentedJSON(http.StatusOK, a)
			return
		}
	}
	c.IndentedJSON(http.StatusNotFound, gin.H{"message": "album not found"})
}
