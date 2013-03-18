package salary

import java.io.FileWriter
import java.io.StringReader

import scala.collection.immutable
import scala.collection.mutable
import scala.util.Random

import au.com.bytecode.opencsv.CSVWriter
import com.mongodb.casbah.Imports._
import edu.stanford.nlp.process.CoreLabelTokenFactory
import edu.stanford.nlp.process.PTBTokenizer

import salary.db.DatabaseManager

object SalaryMain {

  def main(args: Array[String]) = {
    val manager = new DatabaseManager()
    val train = manager.salaryDB("train")
    val test = manager.salaryDB("test")
    val trainFullDescCounter = manager.salaryDB("train_fulldesc_counter")
    val trainFullDescAggregator = manager.salaryDB("train_fulldesc_aggregator")
    val trainFullDescDocFreq = manager.salaryDB("train_fulldesc_docfreq")
    val testFullDescCounter = manager.salaryDB("test_fulldesc_counter")
    val testFullDescAggregator = manager.salaryDB("test_fulldesc_aggregator")
    val testFullDescDocFreq = manager.salaryDB("test_fulldesc_docfreq")

    val javascript = manager.salaryDB("javascript")
    val jsObj = javascript.findOne.get

    generateWords(train, "FullDescription", trainFullDescCounter)
    reduceWords(trainFullDescCounter, trainFullDescAggregator, jsObj)
    reduceDocFreq(trainFullDescCounter, trainFullDescDocFreq, jsObj)

    generateWords(test, "FullDescription", testFullDescCounter)
    reduceWords(testFullDescCounter, testFullDescAggregator, jsObj)
    reduceDocFreq(testFullDescCounter, testFullDescDocFreq, jsObj)

    val trainIds = new mutable.ArrayBuffer[Long]()
    for (elem <- train)
      trainIds += elem.as[Long]("Id")
    val reverseTrainIds: mutable.Map[Long, Int] = new mutable.HashMap()
    for ((elem, index) <- trainIds.zipWithIndex)
      reverseTrainIds(elem) = index

    //MongoDBObject.empty, MongoDBObject("_id" -> 1)
    val wordSet: mutable.Set[String] = new mutable.HashSet[String]()
    for (elem <- trainFullDescAggregator.find)
      wordSet += elem.as[String]("_id")
    //for (elem <- testFullDescAggregator.find)
    //  wordSet += elem.as[String]("_id")
    val words: Seq[String] = wordSet.toSeq
    words.sortWith(_ < _)
    val reverseWords = createReverseMap(words)

    println(reverseWords.size)

    createTable(trainFullDescCounter, words)
  }

  private[this] def generateWords(inColl: MongoCollection, field: String, outColl: MongoCollection): Unit = {
    if (!outColl.isEmpty)
      return
    val words: mutable.Map[String, Int] = mutable.HashMap().withDefaultValue(0)
    val numElems = inColl.find.count.toDouble
    var index: Long = 0
    var percentDone: Int = 0
    for (elem <- inColl.find) {
      index += 1
      val newPercentDone = (100 * index / numElems).toInt
      if (newPercentDone != percentDone) {
        percentDone = newPercentDone
        println("Progress generating words: %d %%".format(percentDone))
      }
      val fullDescription = elem(field)
      val ptbt = new PTBTokenizer(new StringReader(fullDescription.toString),
        new CoreLabelTokenFactory(), "")
      words.clear
      while (ptbt.hasNext) {
        val coreLabel = ptbt.next
        val word = coreLabel.word
        words(word) += 1
      }
      val builder = MongoDBList.newBuilder
      for ((key, value) <- words) {
        val entry = MongoDBObject(
          "word" -> key,
          "counter" -> value)
        builder += entry
      }
      val obj = MongoDBObject(
        "Id" -> elem("Id"),
        "arr" -> builder.result)
      outColl += obj
    }
  }

  private[this] def reduceWords(inColl: MongoCollection, outColl: MongoCollection, jsObj: MongoDBObject): Unit = {
    if (!outColl.isEmpty)
      return
    inColl.mapReduce(
      jsObj("wordsMapFunction").toString,
      jsObj("wordsReduceFunction").toString,
      outColl.name)
  }

  private[this] def reduceDocFreq(inColl: MongoCollection, outColl: MongoCollection, jsObj: MongoDBObject): Unit = {
    if (!outColl.isEmpty)
      return
    inColl.mapReduce(
      jsObj("docfreqMapFunction").toString,
      jsObj("docfreqReduceFunction").toString,
      outColl.name)
  }

  private[this] def createTable(inColl: MongoCollection, words: Seq[String]): Unit = {
    val shuffledWords: Seq[String] = Random.shuffle(words)
    val reverseShuffledWords = createReverseMap(shuffledWords)
    val numChunks = 2931
    val chunkSize = shuffledWords.size / numChunks

    val writers = new Array[CSVWriter](numChunks)
    for (i <- List.range(0, writers.length))
      writers(i) = new CSVWriter(new FileWriter("data/table/chunk%04d.csv".format(i)))

    try {
      val r = shuffledWords.size % numChunks
      val separators: java.util.List[Integer] = new java.util.ArrayList()
      var left = 0
      separators.add(left)
      for (i <- List.range(0, writers.length)) {
        val offset = if (i < r) 1 else 0
        val right = left + chunkSize + offset
        val chunk = shuffledWords.slice(left, right)
        writers(i).writeNext(chunk.toArray)
        left = right
        separators.add(left)
      }

      val chunkSets = new Array[mutable.TreeSet[(Int, Int)]](numChunks)
      for (i <- List.range(0, numChunks))
        chunkSets(i) = new mutable.TreeSet()

      var numWords = inColl.find.count
      var wordNo: Int = 0
      for (entry <- inColl.find) {
        wordNo += 1
        if (wordNo % 2500 == 0) {
          val percent = 100 * wordNo.toDouble / numWords
          println("Processing chunks: %g percent".format(percent))
        }
        for (i <- List.range(0, chunkSets.size))
          chunkSets(i).clear
        val arr = entry.as[MongoDBList]("arr")
        for (obj <- arr) {
          val dbObj = obj.asInstanceOf[BasicDBObject]
          val word = dbObj.as[String]("word")
          val counter = dbObj.as[Int]("counter")
          val index = reverseShuffledWords(word)
          var pos = java.util.Collections.binarySearch[Integer](separators, index)
          if (pos < 0)
            pos = -pos - 2
          chunkSets(pos) += index -> (index - separators.get(pos))
        }
        for ((chunkSet, i) <- chunkSets.zipWithIndex) {
          val line = new Array[String](2 * chunkSet.size)
          for ((elem, j) <- chunkSet.zipWithIndex) {
            line(2 * j) = elem._1.toString
            line(2 * j + 1) = elem._2.toString
          }
          writers(i).writeNext(line)
        }
      }
    } finally {
      for (writer <- writers)
        writer.close
    }
  }

  private[this] def createReverseMap[T](seq: Seq[T]) = {
    val reverseSeq: mutable.Map[T, Int] = new mutable.HashMap()
    for ((elem, index) <- seq.zipWithIndex)
      reverseSeq(elem) = index
    reverseSeq
  }

}
