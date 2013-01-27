name := "ClickPrediction Leen"

version := "1.0"

scalacOptions ++= Seq("-deprecation", "-unchecked")

libraryDependencies  ++= Seq(
            "org.scalanlp" %% "breeze-math" % "0.1",
            "com.github.scala-incubator.io" %% "scala-io-core" % "0.4.1",
            "com.github.scala-incubator.io" %% "scala-io-file" % "0.4.1"
			)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.2-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
            "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
            "Sonatype tools" at "https://oss.sonatype.org/content/groups/scala-tools/"
            )

mainClass in (Compile, run) := Some("main.scala.sgd.SGD")